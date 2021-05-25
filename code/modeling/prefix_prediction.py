
import sys
import gzip
import json
import pickle
import random
import click
import numpy as np

from tqdm import tqdm
from itertools import chain
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from lightgbm import LGBMClassifier

sys.path.append("../processing/")

from _config import Config



def load_data(ds_name, prefix, outcome, selected_feature_sets):    
    conf = Config()
    ds_x_path = f"{conf.modeling_dir}/prefix/datasets/{ds_name}_p{prefix}.pkl.gz"
    ds_y_path = f"{conf.modeling_dir}/prefix/datasets/{ds_name}_labels.pkl.gz"

    ds_x = pickle.load(gzip.open(ds_x_path))
    ds_y = pickle.load(gzip.open(ds_y_path))
    
    col_idxs = []
    row_idxs = []
    ys = []
    meta = []
    feature_names = []
    
    # select columns
    for idx, feature_pair in enumerate(ds_x["feature_set_name_pairs"]):
        if feature_pair[0] in selected_feature_sets:
            col_idxs.append(idx)
            feature_names.append(feature_pair)
    
    # fetch ys & metadata
    y_key = f"p{prefix}__{outcome}"

    for idx, root_tweet_id in enumerate(ds_x["root_tweet_ids"]):
        # NB: this can happen only for prefix=10
        # as some convs may have < 2*p tweets
        if root_tweet_id not in ds_y:
            continue
            
        if y_key in ds_y[root_tweet_id]:

            conv_dict = ds_y[root_tweet_id]

            row_idxs.append(idx)

            y = conv_dict[y_key]
            ys.append(float(y))

            meta.append({
                "root_tweet_id": conv_dict["root_tweet_id"],
                "root_tweet_type": conv_dict["root_tweet_type"],
                "n": conv_dict["n"],
                "pre_n_tox": conv_dict[f"p{prefix}_pre_n_tox"],
                "suf_n": conv_dict[f"p{prefix}_suf_n"],
                "suf_i_tox": conv_dict[f"p{prefix}_suf_i_tox"],
                "suf_f_tox": conv_dict[f"p{prefix}_suf_f_tox"],
            })
    
    # prepare numpy objs
    X = ds_x["X"]
    X = X[:, col_idxs]
    X = X[row_idxs, :]

    ys = np.array(ys)

    assert X.shape[0] == ys.shape[0]
    
    return X, ys, meta, feature_names


def sanitize_x(X, fnames):
    # remove columns with:
    # 1. >95% missing values
    # 2. no variance
    # 3. corr = 1 with other columns
    
    max_f_nans = 0.95
    
    Xcp = np.copy(X)
    fnames_new = np.array(fnames)
    fnames_removed = []
    
    # (1) remove columns with > 95% missing values
    f_nans = np.sum(np.isnan(Xcp), axis=0) / Xcp.shape[0]
    col_nans = np.flatnonzero(f_nans >= max_f_nans)

    fnames_removed += list(fnames_new[col_nans])
    fnames_new = np.delete(fnames_new, col_nans, axis=0)
    Xcp = np.delete(Xcp, col_nans, axis=1)
    
    # (2) remove columns with no variance
    stds = np.nanstd(Xcp, axis=0)
    no_var = np.flatnonzero(stds == 0)

    fnames_removed += list(fnames_new[no_var])
    fnames_new = np.delete(fnames_new, no_var, axis=0)
    Xcp = np.delete(Xcp, no_var, axis=1)
    
    # (3) remove redundent columns (i.e., corr=1)
    Xcp_i = SimpleImputer().fit_transform(Xcp)  # nan => mean
    rho = np.corrcoef(Xcp_i, rowvar=False)
    rho = np.triu(np.abs(rho), k=1) # abs values & above diagonal
    rho_1 = np.nonzero(rho == 1.0)  
    rho_1_idxs = rho_1[0]           # remove just the first idx
    
    fnames_removed += list(fnames_new[rho_1_idxs])
    fnames_new = np.delete(fnames_new, rho_1_idxs, axis=0)
    Xcp = np.delete(Xcp, rho_1_idxs, axis=1)
    
    # sanity checks
    assert Xcp.shape[1] == fnames_new.shape[0]
    assert np.max(np.sum(np.isnan(Xcp), axis=0) / Xcp.shape[0]) < max_f_nans
    assert np.sum(np.nanstd(Xcp, axis=0) == 0) == 0
    Xcp_i =  SimpleImputer().fit_transform(Xcp)
    rho_abs = np.abs(np.corrcoef(Xcp_i, rowvar=False))
    assert np.sum(np.triu(rho_abs, k=1) == 1.0) == 0
    
    # fnames as lists of lists
    fnames_new = fnames_new.tolist()
    fnames_removed = [i.tolist() for i in fnames_removed]    
    
    return Xcp, fnames_new, fnames_removed


def make_classifiers(cv_n_folds, cv_metric="accuracy"):
    # param grids
    LR_grid = {
        "clf__C": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
    }
    GB_grid = {
        "clf__n_estimators": [10, 25, 50, 100, 500, 1000, 2000, 3000, 5000, 10000]
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


def run_batch(params):
    # fetch parameters
    dataset = params["dataset"]
    prefix = params["prefix"]
    f_groups_str = params["f_groups_str"]
    f_groups_lst = params["f_groups_lst"]
    y_str = params["y_str"]
    outer_n_folds = params["outer_n_folds"]
    inner_n_folds = params["inner_n_folds"]

    metrics = ["accuracy", "roc_auc", "f1", "precision", "recall"]

    X, y, meta, fnames = load_data(dataset, prefix, y_str, f_groups_lst)

    X, fnames_new, fnames_removed = sanitize_x(X, fnames)

    # make and freeze folds s.t. all models run on the same splits
    skf = StratifiedKFold(n_splits=outer_n_folds, shuffle=True, random_state=0)
    skf_lst = list(skf.split(X, y))

    clf_pairs = make_classifiers(inner_n_folds, "accuracy")

    # shuffle classifiers to avoid running high cpu/mem classifiers together
    # random.shuffle(clf_pairs)

    all_res = []

    for clf_name, clf in clf_pairs:
        print(f">> {dataset} | {prefix} | {f_groups_str} | {clf_name}")
        
        res = {
            "dataset": dataset,
            "prefix": prefix,
            "outcome": y_str,
            "n_samples": X.shape[0],
            "outer_n_folds": outer_n_folds,
            "inner_n_folds": inner_n_folds,
            "feature_groups": f_groups_str,
            "features": fnames_new,
            "features_excluded": fnames_removed,
            "clf": clf_name
        }
        
        cv_res = cross_validate(
            clf,
            X,
            y,
            cv=skf_lst,
            scoring=metrics,
            return_train_score=True,
            return_estimator=True,
            n_jobs=10
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
    
    prefixes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    outer_n_folds = 10
    inner_n_folds = 5

    feature_sets = [
        "tree",
        "follow_graph",
        "reply_graph",
        "embeddedness",
        "polarization",
        "subgraph",
        "arrival_seq",
        "rate",
        "toxicity"
    ]

    feature_sets_no_tox = feature_sets[:]
    feature_sets_no_tox.remove("toxicity")

    # NB: 11 groups in total
    feature_groups = ([("all", feature_sets)] + 
        [("all/toxicity", feature_sets_no_tox)] +
        [(fs, [fs]) for fs in feature_sets])
    
    # outcome
    y_str = "suf_f_tox__tox_bucket__>=q50"

    # make param dictionaries
    batches_params = []

    for prefix in prefixes:
        for f_groups in feature_groups:
            batches_params.append({
                "dataset": dataset,
                "prefix": prefix,
                "outer_n_folds": outer_n_folds,
                "inner_n_folds": inner_n_folds,
                "f_groups_str": f_groups[0],
                "f_groups_lst": f_groups[1],
                "y_str": y_str
            })

    print(f"Number of batches: {len(batches_params)}")
    
    # run batches
    parallel = Parallel(n_jobs=n_jobs, verbose=10)
    res_lists = parallel(delayed(run_batch)(params) for params in batches_params)

    # flatten lists => single list of dicts
    res = list(chain.from_iterable(res_lists))
    
    print(f"|results| = {len(res)}")
    
    # write results to JSON
    out_dir = f"{Config().modeling_dir}/prefix/runs"
    out_fpath = f"{out_dir}/{dataset}_q50_nested_cv.json.gz"

    with gzip.open(out_fpath, "wt") as fout:
        json.dump(res, fout)
    
    print("Done!")


if __name__ == "__main__":
    main()

# END