
import gzip
import click
import warnings
import json
import ujson
import pickle
import numpy as np
from tqdm import tqdm 
from ciso8601 import parse_datetime
from joblib import Parallel, delayed

from _config import Config
from utils import json_paths_iter, sanitize_numpy_types

# metrics methods
from tree_metrics import tree_metrics
from follow_graph_metrics import follow_graph_metrics
from reply_graph_metrics import reply_graph_metrics
from emb_metrics import embeddedness_metrics
from polarization_metrics import polarization_metrics
from subgraph_metrics import get_conversation_census as subgraph_metrics



def toxicity_metrics(conversation):
    tweet_ids = conversation["tweets"].keys()
    tox_scores_dict = conversation["toxicity_scores"]

    # fetch toxicity scores 
    tox_scores = [tox_scores_dict[t_id] \
                  for t_id in tweet_ids if t_id in tox_scores_dict]
    tox_scores = np.asarray(tox_scores)

    # compute statistics
    if len(tox_scores) > 0:
        mean = float(np.mean(tox_scores))

        std = float(np.std(tox_scores))

        quartiles = (np
            .quantile(tox_scores, [0, 0.25, 0.5, 0.75, 1])
            .tolist())

        quartiles = dict(zip(("q0", "q25", "q50", "q75", "q100"), quartiles))

        metrics = {"mean": mean, "std": std, **quartiles}

    else:
        m_names = ["mean", "std", "q0", "q25", "q50", "q75", "q100"]
        metrics = {i: None for i in m_names}

    return metrics


def arrival_metrics(conversation):
    # NB: indicies range from 1 ... n
    tweets = list(conversation["tweets"].values())
    tweets.sort(key=lambda x: x["time"])

    user_id_to_seq_id = {}
    max_seq_id = 0

    user_seq_ids = []
    user_n_uniq = []

    for tweet in tweets:
        u_id = tweet["user_id"]
    
        if u_id not in user_id_to_seq_id:
            user_id_to_seq_id[u_id] = max_seq_id
            max_seq_id += 1

        u_seq_id = user_id_to_seq_id[u_id]

        user_seq_ids.append(u_seq_id)
        user_n_uniq.append(max_seq_id)

    n = len(user_seq_ids)
    user_seq_ids_dict = {f"user_seq_id_{i}": user_seq_ids[i] for i in range(1, n)}
    user_n_uniq_dict = {f"n_unique_{i}": user_n_uniq[i] for i in range(1, n)}

    metrics = {**user_seq_ids_dict, **user_n_uniq_dict}
    
    return metrics


def rate_metrics(conversation):
    # NB: There are:
    # - n-1 times, excluding the root [1 ... n-1]
    # - n-2 deltas, excluding the time between the root and the first
    #       reply because that is equal to time_1 [2 ... n-1]

    tweets = list(conversation["tweets"].values())
    tweets.sort(key=lambda x: x["time"])

    n = len(tweets)
    
    # # [NB: already normized the tweet times during anonymization]
    # # parse time
    # times_abs = [parse_datetime(tweet["time"]) for tweet in tweets]
    
    # # compute relative times
    # t0 = times_abs[0]
    # times = [(t - t0).total_seconds() for t in times_abs]
    
    # relative times [on anonymized data]
    times = [tweet["time"] for tweet in tweets]

    # compute time deltas
    times_d = [times[i] - times[i - 1] for i in range(1, n)]
    
    # agg stats
    n_d_half = int(round(len(times_d) / 2))
    times_d_arr = np.asarray(times_d)

    times_d_mean = np.mean(times_d_arr)
    times_d_1half_mean = np.mean(times_d_arr[:n_d_half])
    times_d_2half_mean = np.mean(times_d_arr[n_d_half:])
    
    # sanity check
    assert np.all(times_d_arr >= 0)
    assert np.all(np.asarray(times) >= 0)
    
    times_dict = {f"time_{i}": times[i] for i in range(1, n)}
    times_d_dict = {f"time_d_{i+1}": times_d[i] for i in range(1, n-1)}

    metrics = {
        **times_dict,
        **times_d_dict,
        "times_d_mean": times_d_mean,
        "times_d_1half_mean": times_d_1half_mean,
        "times_d_2half_mean": times_d_2half_mean
    }
    
    return metrics


def compute_all_metrics(conversation):
    # NB:
    # - tree metrics: n_tweets does NOT include the root 
    
    # params
    tree_depth_n_levels = 10
    
    metrics = {
        "tree": tree_metrics(conversation, tree_depth_n_levels),
        "follow_graph": follow_graph_metrics(conversation, remove_root=False),
        "reply_graph": reply_graph_metrics(conversation, remove_root=False),
        "embeddedness": embeddedness_metrics(conversation, remove_root=False),
        "polarization": polarization_metrics(conversation, remove_root=False),
        "subgraph": subgraph_metrics(conversation, remove_root=False),
        "arrival_seq": arrival_metrics(conversation),
        "rate": rate_metrics(conversation),
        "toxicity": toxicity_metrics(conversation)
    }
    
    return metrics


def filter_conversation(conversation, prefix_n):
    # NB: 
    # - returning a deep copy 
    # - removing the body of the tweets to reduce memory footprint
    
    assert prefix_n > 0
    
    # sort tweets in chronological order
    tweets = list(conversation["tweets"].values())
    tweets.sort(key=lambda x: x["time"])

    # filter tweets => contruct tweets_dict, tweet ids, user ids
    pre_tweet_dicts = {}
    pre_tweet_ids = set()
    pre_user_ids = set()

    for tweet in tweets[:prefix_n]:
        t_id = tweet["id"]
        u_id = tweet["user_id"]

        pre_tweet_dicts[t_id] = {
            "id": t_id,
            "user_id": u_id,
            "time": tweet["time"]
        }
        pre_tweet_ids.add(t_id)
        pre_user_ids.add(u_id)

    # filter tree
    pre_tree = filter_tree(conversation["reply_tree"], pre_tweet_ids)

    # filter network features
    pre_network_features = filter_network_features(
        conversation["network_features"], 
        pre_user_ids
    )

    # filter alignments
    pre_alignments = {}

    for u_id, alg in conversation["alignment_scores"].items():
        if u_id in pre_user_ids:
            pre_alignments[u_id] = alg

    # filter toxicity scores
    pre_toxicity_scores = {}

    for t_id, score in conversation["toxicity_scores"].items():
        if t_id in pre_tweet_ids:
            pre_toxicity_scores[t_id] = score

    pre_conversation = {
        "reply_tree": pre_tree,
        "tweets": pre_tweet_dicts,
        "network_features": pre_network_features,
        "alignment_scores": pre_alignments,
        "toxicity_scores": pre_toxicity_scores,
        "root_tweet_type": conversation["root_tweet_type"]
    }

    return pre_conversation


def filter_tree(tree_json, selected_tweet_ids):
    tweet_id = tree_json["tweet"]
    tweet_replies = tree_json["replies"]
    
    # stopping case
    if tweet_id not in selected_tweet_ids:
        return None
    
    # recursion
    filtered_replies = []

    for subtree in tweet_replies:
        filtered_subtree = filter_tree(subtree, selected_tweet_ids)

        if filtered_subtree is not None:
            filtered_replies.append(filtered_subtree)

    return {"tweet": tweet_id, "replies": filtered_replies}


def filter_network_features(net_features, selected_user_ids):
    # NB: selected_user_ids is a list of strings
    
    # fetch values
    user_ids = net_features["user_ids"]                 # [u_id_int1, ...]
    missing_user_ids = net_features["missing_user_ids"] # [u_id_int1, ...]
    n_friends = net_features["n_friends"]               # [int1, int2, ....]
    n_followers = net_features["n_followers"]           # [int1, int2, ....]
    network = net_features["network"]                   # [[u_idx1, u_idx2],...]
    embs = net_features["network_intersections"]        # [[u_idx1, u_idx2,n_c]] 

    # user_ids / n_friends / n_followers
    new_user_ids = []
    new_n_friends = []
    new_n_followers = []

    idx_old_to_new = {}   # old_idx => new_idx
    new_idx = 0

    for old_idx in range(len(user_ids)):
        if str(user_ids[old_idx]) not in selected_user_ids:
            continue

        new_user_ids.append(user_ids[old_idx])
        new_n_friends.append(n_friends[old_idx])
        new_n_followers.append(n_followers[old_idx])

        idx_old_to_new[old_idx] = new_idx
        new_idx += 1

    # follow network
    new_network = []

    for i_old_idx, j_old_idx in network:
        if i_old_idx in idx_old_to_new and j_old_idx in idx_old_to_new:
            new_network.append([
                idx_old_to_new[i_old_idx], 
                idx_old_to_new[j_old_idx]
            ])

    # embeddedness
    new_embs = []

    for i_old_idx, j_old_idx, n_common in embs:
        if i_old_idx in idx_old_to_new and j_old_idx in idx_old_to_new:
            new_embs.append([
                idx_old_to_new[i_old_idx], 
                idx_old_to_new[j_old_idx],
                n_common
            ])

    # filter missing user_ids
    new_missing_user_ids = [u_id for u_id in missing_user_ids \
            if str(u_id) in selected_user_ids]

    new_net_features = {
        "user_ids": new_user_ids,
        "missing_user_ids": new_missing_user_ids,
        "n_friends": new_n_friends,
        "n_followers": new_n_followers,
        "network": new_network,
        "network_intersections": new_embs
    }

    return new_net_features


def conversation_prefix_metrics(json_fpath, prefix_sizes):
    # Small trick to save memory:
    # - filtering each conversation from the largest to the smallest prefix
    # - computing the prefix incrementally (from the previous iteration)
    #   rather than the full conversation
    
    # there is one division with zero warning that is actually handled
    warnings.filterwarnings("ignore")
    
    conversation = ujson.load(gzip.open(json_fpath))
    n_tweets = len(conversation["tweets"])
    
    prefix_metrics = {}
    prefix_sizes = sorted(prefix_sizes, reverse=True)
    
    for prefix_size in prefix_sizes:
        if n_tweets > prefix_size:
            conversation = filter_conversation(conversation, prefix_size)
            prefix_metrics[prefix_size] = compute_all_metrics(conversation)
    
    if len(prefix_metrics) > 0:
        prefix_metrics["root_tweet_id"] = conversation["reply_tree"]["tweet"]
    
    return prefix_metrics


def compute_prefix_metrics(dataset, n_jobs=1, limit=None):

    prefixes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    print("--- Prefix Metrics ---")
    print(f"Dataset: {dataset}")
    print(f"Num Jobs: {n_jobs}")
    print(f"Limit: {limit}")
    print(f"Prefixes: {prefixes}")
    print("----------------------------")
    
    # paths
    conf = Config(dataset)
    
    output_fpath = f"{conf.data_root}/prefix_metrics/{dataset}.json.gz"
    output_pickle_fpath = f"{conf.data_root}/prefix_metrics/{dataset}.pkl.gz"

    json_fpaths = json_paths_iter(conf.conversations_jsons_dir, limit=limit)

    # compute metrics
    print("Computing metrics ...")

    if n_jobs == 1:
        metrics = [conversation_prefix_metrics(json_fpath, prefixes) \
            for json_fpath in tqdm(json_fpaths)]
    else:
        parallel = Parallel(n_jobs=n_jobs, verbose=10)
        metrics = parallel(
            delayed(conversation_prefix_metrics)(json_fpath, prefixes) \
                for json_fpath in json_fpaths
            )

    print(f"Metrics total: {len(metrics)}")

    # skip empty results
    metrics = [m for m in metrics if len(m) > 0]
    print(f"Metrics non-zero: {len(metrics)}")

    # pickle
    with gzip.open(output_pickle_fpath, "wb") as fout:
        pickle.dump(metrics, fout, protocol=4)

    # uJSON complains: cast numpy ints/floats to python ints/floats 
    for conv_metrics in metrics:
        for prefix_n, prefix_metrics in conv_metrics.items():
            if prefix_n != "root_tweet_id":
                for group_name, group_values in prefix_metrics.items():
                    if group_values is not None:
                        group_values = sanitize_numpy_types(group_values)

    # output metrics to JSON
    print("Outputting results to JSON ...")
    with gzip.open(output_fpath, "wt") as fout:
        json.dump(metrics, fout)

    print("Done!")


@click.command()
@click.option('--dataset', required=True,
    type=click.Choice(["news", "midterms"]))
@click.option('--n_jobs', required=True, type=int)
@click.option('--limit', default=None, type=int)
def main(dataset, n_jobs, limit):
    
    compute_prefix_metrics(
        dataset, 
        n_jobs=n_jobs, 
        limit=limit
    )


if __name__ == "__main__":
    main()

# END
