
import gzip
import json
import numpy as np

from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

sys.path.append("../processing/")

from _config import Config
from prediction_prefix import load_data, sanitize_x


def get_GB_model(n_folds, n_jobs):
    grid = {
        "clf__n_estimators": [10, 25, 50, 100, 500, 1000, 2000, 3000, 5000, 10000]
    }

    pipe = Pipeline([
        ("imp", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("std", StandardScaler(copy=True, with_mean=True, with_std=True)),
        ("clf", LGBMClassifier(random_state=0))
    ])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=0)

    model = GridSearchCV(
        estimator=pipe, 
        param_grid=grid, 
        cv=skf,
        n_jobs=n_jobs,
        scoring="accuracy",
        refit=True
    )
    
    return model


def run(prefix, cv_n_folds, cv_n_jobs):
    print(f">> Prefix: {prefix}")

    # common settings
    y_str = "suf_f_tox__tox_bucket__>=q50"
    f_groups_lst = [
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

    # load data
    X_nw, y_nw, _, fnames_nw = load_data("news", prefix, y_str, f_groups_lst)
    X_mt, y_mt, _, fnames_mt = load_data("midterms", prefix, y_str, f_groups_lst)

    assert fnames_nw == fnames_mt
    fnames = fnames_nw
    
    # sanitize
    _, fnames_new_nw, _ = sanitize_x(X_nw, fnames_nw)
    _, fnames_new_mt, _ = sanitize_x(X_mt, fnames_mt)
    
    # intersect features
    fnames_new_nw = [tuple(i) for i in fnames_new_nw]
    fnames_new_mt = [tuple(i) for i in fnames_new_mt]

    fnames_int = set(fnames_new_nw) & set(fnames_new_mt)
    fnames_int_idxs = sorted([fnames.index(common) for common in fnames_int])

    # filter
    X_nw = X_nw[:, fnames_int_idxs]
    X_mt = X_mt[:, fnames_int_idxs]

    assert X_nw.shape[1] == X_mt.shape[1]    

    # train / test split
    X_nw_train, X_nw_test, y_nw_train, y_nw_test = \
        train_test_split(X_nw, y_nw, test_size=0.2, random_state=0)
    X_mt_train, X_mt_test, y_mt_train, y_mt_test = \
        train_test_split(X_mt, y_mt, test_size=0.2, random_state=0)

    # news model
    model_nw = get_GB_model(n_folds=cv_n_folds, n_jobs=cv_n_jobs)
    model_nw.fit(X_nw_train, y_nw_train)

    # midterms model
    model_mt = get_GB_model(n_folds=cv_n_folds, n_jobs=cv_n_jobs)
    model_mt.fit(X_mt_train, y_mt_train)

    res = {
        "prefix": prefix,
        # news
        "news_train": accuracy_score(model_nw.predict(X_nw_train), y_nw_train),
        "news_news_test": accuracy_score(model_nw.predict(X_nw_test), y_nw_test),
        "news_midterms_test": accuracy_score(model_nw.predict(X_mt_test), y_mt_test),
        # midterms
        "midterms_train": accuracy_score(model_mt.predict(X_mt_train), y_mt_train),
        "midterms_midterms_test": accuracy_score(model_mt.predict(X_mt_test), y_mt_test),
        "midterms_news_test": accuracy_score(model_mt.predict(X_nw_test), y_nw_test)       
    }
    
    return res


def main():
    main_n_jobs = 4
    cv_n_jobs = 10
    cv_n_folds = 10
    prefixes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    parallel = Parallel(n_jobs=main_n_jobs, verbose=10)
    res = parallel(
        delayed(run)(
            prefix, 
            cv_n_folds, 
            cv_n_jobs
        ) 
        for prefix in prefixes
    )

    # print(res)

    # write results to JSON
    out_fpath = f"{Config().modeling_dir}/prefix/runs/domain_transfer.json.gz"

    with gzip.open(out_fpath, "wt") as fout:
        json.dump(res, fout)
    
    print("Done!")


if __name__ == "__main__":
    main()

# END