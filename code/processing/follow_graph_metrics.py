
import gzip
import ujson as json
import click
from tqdm import tqdm
from joblib import Parallel, delayed

from _config import Config
from utils import json_paths_iter, sanitize_numpy_types, write_dicts_to_csv
from graph_conversions import follow_json_to_nx_graph
from graph_metrics import compute_undirected_graph_metrics
from graph_metrics import compute_directed_graph_metrics
from graph_metrics import n_friends_followers_stats



def follow_graph_metrics(conversation, remove_root):
    
    net_features = conversation["network_features"]
    
    # fetch root info
    root_tweet_id = conversation["reply_tree"]["tweet"]
    root_user_id = conversation["tweets"][root_tweet_id]["user_id"]
    
    # NB: sometimes the root might be missing => no need to remove it then
    if remove_root and root_user_id in net_features["missing_user_ids"]:
        remove_root = False
    
    # remove root => root_user_id = None
    if not remove_root:
        root_user_id = None
    
    # create networkx graphs
    G_di = follow_json_to_nx_graph(
        conversation, 
        directed=True, 
        remove_root=remove_root
    )
    G_ud = follow_json_to_nx_graph(
        conversation, 
        directed=False, 
        remove_root=remove_root
    )
    
    # sanity check
    n_users = len(net_features["user_ids"]) - int(remove_root)
    assert(len(G_di) == len(G_ud) == n_users)
    
    # NB: skip conversations with 0 nodes, 
    #     this can happen if there only tweets by the root
    #     and we want to remove the root.
    if len(G_di) == 0:
        return None
    
    # undirected graph metrics
    g_ud_metrics = compute_undirected_graph_metrics(G_ud)    
    
    # directed graph metrics
    g_di_metrics = compute_directed_graph_metrics(G_di)

    # friends and followers stats
    n_friends_followers_metrics = n_friends_followers_stats(
        G_di, 
        net_features, 
        root_user_id
    )
        
    metrics = {
        "root_tweet_id": root_tweet_id,
        "n_nodes": len(G_di),
        **g_di_metrics,
        **g_ud_metrics,
        **n_friends_followers_metrics
    }

    # convert numpy types to python types
    metrics = sanitize_numpy_types(metrics)

    return metrics


def compute_metrics(json_fpath):
    conversation = json.load(gzip.open(json_fpath))
    metrics = follow_graph_metrics(conversation, remove_root=False)
    return metrics


#
# FOLLOW GRAPH METRICS COMPUTATION
#
def compute_follow_graph_metrics(dataset, n_jobs=1, limit=None):

    print("--- Follow Graph Metrics ---")
    print(f"Dataset: {dataset}")
    print(f"Num Jobs: {n_jobs}")
    print(f"Limit: {limit}")
    print("----------------------------")
    
    # paths
    conf = Config(dataset)
    
    output_fpath = f"{conf.data_root}/follow_graph_metrics.csv"

    # iterator
    json_fpaths = json_paths_iter(
        conf.conversations_no_embs_jsons_dir, 
        limit=limit
    )

    # compute metrics
    print("Computing metrics ...")

    if n_jobs == 1:
        metrics = [compute_metrics(json_fpath) \
            for json_fpath in tqdm(json_fpaths)]
    else:
        parallel = Parallel(n_jobs=n_jobs, verbose=10)
        metrics = parallel(
            delayed(compute_metrics)(json_fpath) \
                for json_fpath in json_fpaths
            )

    print("Output:", len(metrics))

    # output to csv
    print("Outputting follow graph metrics to CSV ...")
    write_dicts_to_csv(metrics, output_fpath)

    print("Done!")


@click.command()
@click.option('--dataset', required=True,
    type=click.Choice(["news", "midterms"]))
@click.option('--n_jobs', required=True, type=int)
@click.option('--limit', default=None, type=int)
def main(dataset, n_jobs, limit):

    compute_follow_graph_metrics(
        dataset, 
        n_jobs=n_jobs, 
        limit=limit
    )


if __name__ == "__main__":
    main()

# END
