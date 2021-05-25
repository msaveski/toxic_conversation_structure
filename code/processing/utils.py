
import os
import csv
import numpy as np


def json_paths_iter(dir_path, limit=None):
    # NB: sort the files so that the order is consistent across runs
    fnames = os.listdir(dir_path)
    fnames.sort()

    for i, fname in enumerate(fnames):
        if limit is not None and i >= limit:
            break
        yield dir_path + fname


def pickle_paths_iter(dir_path, N, limit=None, to_skip=[]):
    for i in range(N):
        if limit is not None and i >= limit:
            break
        if i in to_skip:
            continue
        yield f"{dir_path}/{i}.pkl"


def sanitize_numpy_types(metrics):
    # np.int64 => int
    # np.float64 => float
    
    for m, v in metrics.items():
        if isinstance(v, np.int64):
            v = int(v)
        if isinstance(v, np.float64):
            v = float(v)
        metrics[m] = v
        
    return metrics


def write_dicts_to_csv(dicts, output_fpath):
    # remove Nones
    dicts_no_nones = [d for d in dicts if d is not None]
    n_nones = len(dicts) - len(dicts_no_nones)

    if n_nones > 0:
        print("[CSV]: Skipped Nones:", n_nones)

    # output to CSV
    fieldnames = list(dicts_no_nones[0].keys())
    with open(output_fpath, "w") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dicts_no_nones)


def is_toxic(toxicity_scores, tweet_id, threshold=0.531):
    if tweet_id not in toxicity_scores:
        return None
    
    return toxicity_scores[tweet_id] > threshold

# END