
import sys
import gzip
import json
import pickle
import random
import click
import numpy as np

from tqdm import tqdm
from itertools import chain
from collections import defaultdict
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

sys.path.append("../processing/")

from _config import Config
from prefix_prediction import sanitize_x



def load_data(ds_name, selected_feature_sets):
    # laod ds
    ds_dir = f"{Config().modeling_dir}/next_reply/datasets"
    ds_fpath = f"{ds_dir}/{ds_name}_paired.pkl.gz"
    ds = pickle.load(gzip.open(ds_fpath))

    X = ds["X"]
    y = ds["y"]
    meta = ds["meta"]

    # select columns and feature names
    col_idxs = []
    feature_names = []

    for idx, feature_pair in enumerate(ds["feature_set_name_pairs"]):
        if feature_pair[0] in selected_feature_sets:
            col_idxs.append(idx)
            feature_names.append(feature_pair)

    assert len(col_idxs) > 0

    # subset features
    X = X[:, col_idxs]

    # sanity checks
    assert X.shape[0] == y.shape[0] == len(meta)
    assert X.shape[1] == len(feature_names)
    assert str(X.dtype) == "float64"
    assert str(y.dtype) == "float64"

    return X, y, meta, feature_names


def make_paired_dataset(X, y, meta):
    # Pair tweets from the same conversation (i.e., root_tweet_id) and 
    # take the difference between their features.
    # Half will be positive examples (i.e., tox tweet - non-tox tweet)
    # and half will be negative examples.
    # Combine the metadata where suitable.
    
    # find the pos (tox) and neg (non-tox) tweet for each root
    # root_id => [tweet_id1, tweet_id2 ...]
    r_pos = defaultdict(list)
    r_neg = defaultdict(list)

    for m in meta:
        tweet_id = m["tweet_id"]
        root_id = m["root_tweet_id"]
        tox_score = m["tox_score"]

        if tox_score > 0.75:
            r_pos[root_id].append(tweet_id)
        elif tox_score < 0.25:
            r_neg[root_id].append(tweet_id)
    
    # pair positive and negative examples
    pairs_t_ids = []

    for root_id in r_pos.keys():
        assert len(r_pos[root_id]) == 1 and len(r_neg[root_id]) == 1    
        pos_t_id = r_pos[root_id][0]
        neg_t_id = r_neg[root_id][0]
        pairs_t_ids.append((pos_t_id, neg_t_id))
    
    # half of the pairs will be positive (i.e., tox tweet goes first)
    n = len(pairs_t_ids)
    n_half = int(n / 2)

    sides = np.zeros(n)
    sides[:n_half] = 1
    
    # shuffle 
    RNG = random.Random(0)
    RNG.shuffle(pairs_t_ids)
    RNG.shuffle(sides)
    
    # tweet idx -> idxs
    tweet_id_to_idx = {m["tweet_id"]: idx for idx, m in enumerate(meta)}

    # compute X (differences of pos & neg) and meta
    X_pairs_lst = []
    meta_pairs = []

    for i in range(n):
        pos_t_id, neg_t_id = pairs_t_ids[i]
        side = sides[i]

        pos_t_idx = tweet_id_to_idx[pos_t_id]
        neg_t_idx = tweet_id_to_idx[neg_t_id]

        # X diff
        x_pos = X[pos_t_idx, :]
        x_neg = X[neg_t_idx, :]

        if side == 1.0:
            x_diff = x_pos - x_neg

        elif side == 0.0:
            x_diff = x_neg - x_pos

        X_pairs_lst.append(x_diff)

        # meta
        meta_pos = meta[pos_t_idx]
        meta_neg = meta[neg_t_idx]    

        assert meta_pos["root_tweet_id"] == meta_neg["root_tweet_id"]
        assert meta_pos["root_tweet_type"] == meta_neg["root_tweet_type"]    

        tox_score_d = meta_pos["tox_score"] - meta_neg["tox_score"]

        meta_pairs.append({
            "root_tweet_id": meta_pos["root_tweet_id"],
            "root_tweet_type": meta_pos["root_tweet_type"],
            "tox_score_abs_delta": np.abs(tox_score_d)
        })

    X_pairs = np.stack(X_pairs_lst, axis=0)
    y_pairs = np.array(sides, copy=True)

    # sanity check
    assert X_pairs.shape[0] == y_pairs.shape[0] == len(meta_pairs)

    return X_pairs, y_pairs, meta_pairs


def make_classifiers(cv_n_folds, cv_metric="accuracy"):

    # param grids
    LR_grid = {
        "clf__C": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
    }
    GB_grid = {
        "clf__n_estimators": [50, 100, 500, 1000, 2000, 3000, 5000, 10000]
    }

    # make pipelines
    LR_pipe = Pipeline([
        ("imp", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("std", StandardScaler(copy=True, with_mean=True, with_std=True)),
        ("clf", LogisticRegression(max_iter=10000, random_state=0))
    ])
    GB_pipe = Pipeline([
        ("imp", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("std", StandardScaler(copy=True, with_mean=True, with_std=True)),
        ("clf", LGBMClassifier(random_state=0))
    ])

    # k-fold split (same for both classifiers)
    LR_skf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True, random_state=0)
    GB_skf = StratifiedKFold(n_splits=cv_n_folds, shuffle=True, random_state=0)

    # make grid search objects
    LR_CV = GridSearchCV(
        estimator=LR_pipe, 
        param_grid=LR_grid, 
        cv=LR_skf,
        scoring=cv_metric,
        refit=True,
    )
    GB_CV = GridSearchCV(
        estimator=GB_pipe, 
        param_grid=GB_grid, 
        cv=GB_skf,
        scoring=cv_metric,
        refit=True
    )

    clf_pairs = [("LR", LR_CV), ("GB", GB_CV)]

    # # <test>
    # dummy_clf = Pipeline([
    #     ("imp", SimpleImputer(missing_values=np.nan, strategy="mean")),
    #     ("std", StandardScaler(copy=True, with_mean=True, with_std=True)),
    #     ("clf", DummyClassifier(strategy="uniform"))
    # ])
    # clf_pairs = [("dummy_uni", dummy_clf)]
    # # </test>

    return clf_pairs


def run_batch(dataset, f_groups_str, f_groups_lst, outer_n_folds, inner_n_folds):
    
    metrics = ["accuracy", "roc_auc", "f1", "precision", "recall"]

    X, y, meta, fnames = load_data(dataset, f_groups_lst)

    X, fnames_new, fnames_removed = sanitize_x(X, fnames)

    X_pairs, y_pairs, meta_pairs = make_paired_dataset(X, y, meta)

    # make and freeze folds s.t. all models run on the same splits
    skf = StratifiedKFold(n_splits=outer_n_folds, shuffle=True, random_state=0)
    skf_lst = list(skf.split(X_pairs, y_pairs))

    clf_pairs = make_classifiers(inner_n_folds, "accuracy")

    # shuffle classifiers to avoid running high cpu/mem classifiers together
    random.shuffle(clf_pairs)

    all_res = []

    for clf_name, clf in clf_pairs:
        print(f">> {dataset} | {f_groups_str} | {clf_name}")

        res = {
            "dataset": dataset,
            "feature_groups": f_groups_str,
            "outer_n_folds": outer_n_folds,
            "inner_n_folds": inner_n_folds,
            "clf": clf_name
        }

        cv_res = cross_validate(
            clf,
            X_pairs,
            y_pairs,
            cv=skf_lst,
            scoring=metrics,
            return_train_score=True,
            return_estimator=True
        )

        # best estimator params
        cv_res_est = [str(est.best_params_) for est in cv_res["estimator"]]
        cv_res["estimator"] = np.array(cv_res_est)

        cv_res = {k: v.tolist() for k, v in cv_res.items()}

        res.update(cv_res)

        all_res.append(res)
        
    return all_res


@click.command()
@click.option('--dataset', required=True, type=click.Choice(["news", "midterms"]))
@click.option('--n_jobs', required=True, type=int)
def main(dataset, n_jobs):
    outer_n_folds = 10
    inner_n_folds = 5

    # feature groups
    all_feature_sets = [
        "conversation_state",
        "user_info",
        "alignments",
        "follow_di",
        "follow_ud",
        "reply_di",
        "reply_ud",
        "dyad_up",
        "dyad_ur",
        "embeddedness_all",
        "embeddedness_toxicity",
        "embeddedness_follow",
        "embeddedness_reply",
        "tree"    
    ]

    feature_groups = [
        ("conversation_state", "conversation_state"),
        ("dyad_up", "dyad_up"),
        ("dyad_ur", "dyad_ur"),
        ("tree", "tree"),
        ("follow_di_ud_emb", ["follow_di", "follow_ud", "embeddedness_follow"]),
        ("reply_di_ud_emb", ["reply_di", "reply_ud", "embeddedness_reply"]),
        ("embeddedness_all", "embeddedness_all"), 
        ("embeddedness_toxicity", "embeddedness_toxicity"), 
        ("alignments", "alignments"), 
        ("user_info", "user_info"), 
        ("all/conversation_state", list(set(all_feature_sets) - set(["conversation_state"]))),
        ("all", all_feature_sets[:])
    ]

    # run batches
    parallel = Parallel(n_jobs=n_jobs, verbose=10)
    res_lists = parallel(
        delayed(run_batch)(
            dataset, 
            f_groups_str, 
            f_groups_lst, 
            outer_n_folds=outer_n_folds, 
            inner_n_folds=inner_n_folds
        ) 
        for f_groups_str, f_groups_lst in feature_groups
    )

    # flatten lists => single list of dicts
    res = list(chain.from_iterable(res_lists))
    
    print(f"|results| = {len(res)}")
    
    # write results to JSON
    out_dir = f"{Config().modeling_dir}/next_reply/runs"
    out_fpath = f"{out_dir}/{dataset}_paired_nested_cv.json.gz"

    with gzip.open(out_fpath, "wt") as fout:
        json.dump(res, fout)
    
    print("Done!")


if __name__ == "__main__":
    main()

# END