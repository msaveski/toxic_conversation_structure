
import gzip
import click
import ujson as json
import numpy as np

from tqdm import tqdm
from joblib import Parallel, delayed

from _config import Config
from utils import json_paths_iter, write_dicts_to_csv



def toxicity_metrics(json_fpath):
    conversation = json.load(gzip.open(json_fpath))
    
    root_tweet_id = conversation["reply_tree"]["tweet"]
    tweet_ids = conversation["tweets"].keys()
    tox_scores_dict = conversation["toxicity_scores"]

    # fetch toxicity scores 
    tox_scores = [tox_scores_dict[t_id] \
                  for t_id in tweet_ids if t_id in tox_scores_dict]
    tox_scores = np.asarray(tox_scores)

    n_scores_missing = len(tweet_ids) - len(tox_scores)
    
    # if none of the tweets have a toxicity score return None
    if len(tox_scores) == 0:
        return None
        
    out = {
        "root_tweet_id": root_tweet_id,
        "n_tweets": len(tweet_ids),
        "n_scores_missing": n_scores_missing,
        "median_toxicity": np.median(tox_scores),
        "mean_toxicity": np.mean(tox_scores)
    }

    # fraction of tweets about certain threasholds
    Ts = [0.25, 0.4, 0.5, 0.531, 0.6, 0.7, 0.75, 0.8, 0.9]

    for T in Ts:
        T_str = str(T).replace(".", "_")
        out[f"f_tox_tweets_{T_str}"] = np.mean(tox_scores > T)

    return out


def compute_toxicity_metrics(dataset, n_jobs=1, limit=None):
    print("--- Toxicity Metrics ---")    
    print(f"Dataset: {dataset}")
    print(f"Num Jobs: {n_jobs}")
    print(f"Limit: {limit}")
    print("----------------------------")

    # paths
    conf = Config(dataset)
    
    output_fpath = f"{conf.data_root}/toxicity.csv"

    # iterator
    json_fpaths = json_paths_iter(
        conf.conversations_no_embs_jsons_dir, 
        limit=limit
    )

    # compute metrics
    print("Computing metrics ...")
    
    if n_jobs == 1:
        metrics = [toxicity_metrics(json_fpath) \
            for json_fpath in tqdm(json_fpaths)]
    else:
        parallel = Parallel(n_jobs=n_jobs, verbose=10)
        metrics = parallel(
            delayed(toxicity_metrics)(json_fpath) \
                for json_fpath in json_fpaths
            )

    print("Metrics computed:", len(metrics))    

    print("Outputting metrics to CSV ...")
    write_dicts_to_csv(metrics, output_fpath)

    print("Done!")


@click.command()
@click.option('--dataset', required=True, 
    type=click.Choice(["news", "midterms"], case_sensitive=False))    
@click.option('--n_jobs', required=True, type=int)
@click.option('--limit', default=None, type=int)
def main(dataset, n_jobs, limit):    
    compute_toxicity_metrics(dataset, n_jobs, limit)


if __name__ == "__main__":
    main()

# END
