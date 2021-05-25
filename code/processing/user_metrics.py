
# NB: Since the user ids are not consistent in the anonymized conversations, 
#   the data to compute the statistics below is not available in the publicly 
#   available dataset.

import gzip
import click
import ujson as json
import numpy as np
from collections import defaultdict
from joblib import Parallel, delayed

from _config import Config
from utils import json_paths_iter, is_toxic, write_dicts_to_csv



def compute_user_conversation_stats(json_fpath, toxicity_threshold=0.531):
    conv = json.load(gzip.open(json_fpath))
    
    tweets = list(conv["tweets"].values())
    tox_scores = conv["toxicity_scores"]
    
    # user_id => {n_tweets: x, n_tox_tweets: x}
    user_conv_stats = {}
    n_tox = 0    
    
    for tweet in tweets:
        t_id, u_id = tweet["id"], tweet["user_id"]
        t_tox = is_toxic(tox_scores, t_id, toxicity_threshold)
        t_tox = t_tox is not None and t_tox == True
        
        u_c_stats = user_conv_stats.get(u_id, {"n_tweets": 0, "n_tox_tweets": 0})        
        
        u_c_stats["n_tweets"] += 1
        u_c_stats["n_tox_tweets"] += int(t_tox)

        user_conv_stats[u_id] = u_c_stats
        n_tox += int(t_tox)
    
    assert sum(uc.get("n_tweets", 0) for uc in user_conv_stats.values()) == len(tweets)
    assert sum(uc.get("n_tox_tweets", 0) for uc in user_conv_stats.values()) == n_tox
    
    return user_conv_stats


def agg_user_stats(all_user_conv_stats):
    # aggregate the user stats across many conversations
    
    user_stats = {}

    for user_conv_stats in all_user_conv_stats:
        for u_id, u_c_stats in user_conv_stats.items():
            u_n_tweets = u_c_stats["n_tweets"]
            u_n_tox_tweets = u_c_stats["n_tox_tweets"]

            u_stats = user_stats.get(u_id, defaultdict(int))

            u_stats["n_tweets"] += u_n_tweets
            u_stats["n_tox_tweets"] += u_n_tox_tweets

            u_stats["n_convs"] += 1
            u_stats["n_tox_convs"] += int(u_n_tox_tweets > 0)

            user_stats[u_id] = u_stats

    for u_stats in user_stats.values():
        u_stats["f_tox_tweets"] = u_stats["n_tox_tweets"] / u_stats["n_tweets"]
        u_stats["f_tox_convs"] = u_stats["n_tox_convs"] / u_stats["n_convs"]
        
    return user_stats


#
# USER METRICS
#
def compute_user_metrics(dataset, n_jobs=1, limit=None):
    print("--- User Metrics ---")
    print(f"Dataset: {dataset}")
    print(f"Num Jobs: {n_jobs}")
    print(f"Limit: {limit}")
    print("----------------------------")
    
    toxicity_threshold = 0.531

    conf = Config(dataset)

    json_fpaths = json_paths_iter(
        conf.conversations_no_embs_jsons_dir, 
        limit=limit
    )

    # all_user_conv_stats = [
    #     compute_user_conversation_stats(json_fpath, toxicity_threshold) \
    #     for json_fpath in json_fpaths]

    parallel = Parallel(n_jobs=n_jobs, verbose=10)
    all_user_conv_stats = parallel(
        delayed(compute_user_conversation_stats)(
                json_fpath,
                toxicity_threshold
            ) \
            for json_fpath in json_fpaths
        )
    
    print("Aggregating user metrics ...")
    user_stats = agg_user_stats(all_user_conv_stats)

    user_stats_csv = [{"user_id": u_id, **u_stats} \
                    for u_id, u_stats in user_stats.items()]

    # out_json_fpath = f"{conf.data_root}/user_metrics.json.gz"
    # json.dump(user_stats, gzip.open(out_json_fpath, "wt"), indent=2)

    out_csv_fpath = f"{conf.data_root}/user_metrics.csv"
    write_dicts_to_csv(user_stats_csv, out_csv_fpath)

    print("Done!")


@click.command()
@click.option('--dataset', required=True, 
    type=click.Choice(["news", "midterms"], case_sensitive=False))
@click.option('--n_jobs', required=True, type=int)
@click.option('--limit', default=None, type=int)
def main(dataset, n_jobs, limit):
    compute_user_metrics(dataset, n_jobs=n_jobs, limit=limit)


if __name__ == "__main__":
    main()

# END
