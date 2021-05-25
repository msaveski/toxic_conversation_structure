
import sys
import gzip
import click
import numpy as np
import ujson as json
from tqdm import tqdm 
from joblib import Parallel, delayed

from _config import Config
from _common import safe, entropy
from utils import json_paths_iter, write_dicts_to_csv
from tree_conversions import get_tree_user_edges

# uncomment this if you reach the stack limit while parsing the trees
# sys.setrecursionlimit(10000000)



def mean_var_sparse(x, n):
    # computing the mean and the variance more efficiently given only the 
    # non-zero values in x and n, the total size of x
    if n > 0:
        x_sum = np.sum(x)
        x_sq_sum = np.sum(np.square(x))
        x_mean = x_sum / n
        x_var = (x_sq_sum / n) - (np.square(x_mean))
        return x_mean, x_var
    return None, None


def get_reply_edges_user_idxs(tree, tweets, net_user_ids):
    # get all user_ids reply edges
    reply_edges_user_ids = get_tree_user_edges(tree, tweets)
    
    # user idx => user id map (some users can be missing => eg. private accs)
    user_id_to_idx = {u_id: u_idx for u_idx, u_id in enumerate(net_user_ids)}
    
    # NB: one pair of user_ids can occur multiple times
    reply_edges = set()
    
    for u_i_id, u_j_id in reply_edges_user_ids:
        # remove self-loops
        if u_i_id == u_j_id:
            continue
        
        # skip users for which we couldn't get their networks
        if u_i_id not in user_id_to_idx or u_j_id not in user_id_to_idx:
            continue
        
        u_i_idx = user_id_to_idx[u_i_id]
        u_j_idx = user_id_to_idx[u_j_id]
        
        reply_edges.add((u_i_idx, u_j_idx))
        
    return reply_edges


def dyads_stats(edges, n_nodes, root_idx):
    """
    Compute the number of nodes that have 1-way, 2-way and no edges.
    Takes O(|edges|).
    
    NB: you need to know the num of nodes as there might be 0-degree nodes
    
    Input:
        - edges: set of (i, j) tuples
        - n_nodes: number nodes in the graph
        - root_idx: index of the root node
            if None => ignore the root
    """
    n_1way_edges = 0
    n_2way_edges_x2 = 0

    for u, v in edges:
        if root_idx is not None and u == root_idx or v == root_idx:
            continue

        if (v, u) in edges:
            n_2way_edges_x2 += 1
        else:
            n_1way_edges += 1
    
    n_pairs = int(n_nodes * (n_nodes - 1) / 2)
    n_2way_edges = int(n_2way_edges_x2 / 2)
    n_no_edges = n_pairs - (n_1way_edges + n_2way_edges)

    out = {
        "0": n_no_edges,
        "1": n_1way_edges,
        "2": n_2way_edges
    }
    
    return out


def embeddedness_metrics(conversation, remove_root):
    
    # fetch vars
    tree = conversation["reply_tree"]
    tweets = conversation["tweets"]
    
    net_features = conversation["network_features"]
    net_user_ids = net_features["user_ids"]
    n_friends = net_features["n_friends"]
    
    # if you need to remove the root fetch root info
    root_idx = None
    if remove_root:
        root_tweet_id = tree["tweet"]
        root_user_id = tweets[root_tweet_id]["user_id"]
        # sometime we may be missing the root's network
        if root_user_id in net_user_ids:
            root_idx = net_user_ids.index(root_user_id)

    # you must compute the number of users ahead of time as some
    # users may not have any reply/follow edges or emb > 0
    n_users = len(net_features["user_ids"])
    
    if root_idx is not None:
        n_users -= 1
    
    # compile user reply edges given the reply tree [(u_id1, u_id2), ...]
    reply_edges = get_reply_edges_user_idxs(tree, tweets, net_user_ids)
    
    # convert the follow edges to a set
    follow_edges = set(tuple(ij) for ij in net_features["network"])
    
    # fetch embeddedness lists
    embs_list = net_features["network_intersections"]
    
    # edge type over to compute embedding metrics
    etype_vals = ["all", 
                  "reply_e0", "reply_e1", "reply_e2",
                  "follow_e0", "follow_e1", "follow_e2"]
    
    # accum vars
    g_n_common = {i: [] for i in etype_vals}
    g_f_common_union = {i: [] for i in etype_vals}
    g_f_common_min = {i: [] for i in etype_vals}
    
    for i, j, n_common in embs_list:
        # ignore the root
        if root_idx is not None and i == root_idx or j == root_idx:
            continue
        
        n_union = n_friends[i] + n_friends[j] - n_common
        f_common_union = n_common / n_union
        f_common_min = n_common / min(n_friends[i], n_friends[j])
        
        # reply edge type
        reply_edge_type = "reply_e0"
        if (i, j) in reply_edges and (j, i) in reply_edges:
            reply_edge_type = "reply_e2"
        elif (i, j) in reply_edges or (j, i) in reply_edges:
            reply_edge_type = "reply_e1"
        
        # follow edge type
        follow_edge_type = "follow_e0"
        if (i, j) in follow_edges and (j, i) in follow_edges:
            follow_edge_type = "follow_e2"
        elif (i, j) in follow_edges or (j, i) in follow_edges:
            follow_edge_type = "follow_e1"
        
        # update overall conversation stats
        g_n_common["all"].append(n_common)
        g_f_common_union["all"].append(f_common_union)
        g_f_common_min["all"].append(f_common_min)
        
        # update reply graph stats
        g_n_common[reply_edge_type].append(n_common)
        g_f_common_union[reply_edge_type].append(f_common_union)
        g_f_common_min[reply_edge_type].append(f_common_min)
        
        # update follow graph stats
        g_n_common[follow_edge_type].append(n_common)
        g_f_common_union[follow_edge_type].append(f_common_union)
        g_f_common_min[follow_edge_type].append(f_common_min)
        
        
    # compute dyad stats
    reply_dyads_stats = dyads_stats(reply_edges, n_users, root_idx)
    follow_dyads_stats = dyads_stats(follow_edges, n_users, root_idx)
    
    emb_etype_freq = {
        "all": int(n_users * (n_users - 1) / 2),
        **{f"reply_e{i}": reply_dyads_stats[str(i)] for i in (0, 1, 2)},
        **{f"follow_e{i}": follow_dyads_stats[str(i)] for i in (0, 1, 2)}
    }
    
    metrics = {}

    metrics["root_tweet_id"] = conversation["reply_tree"]["tweet"]
    
    for etype in etype_vals:
        n_edge_pairs = emb_etype_freq[etype]
        n_edge_pairs_common_1p = len(g_n_common[etype])
        n_common_f1p = None
        if n_edge_pairs > 0:
            n_common_f1p = n_edge_pairs_common_1p / n_edge_pairs

        n_common_arr = np.array(g_n_common[etype])        
        n_common_mean, n_common_var = mean_var_sparse(n_common_arr, n_edge_pairs)
        metrics[f"n_common_{etype}_mean"] = n_common_mean
        metrics[f"n_common_{etype}_var"] = n_common_var
        metrics[f"n_common_{etype}_f1p"] = n_common_f1p
        metrics[f"n_common_{etype}_1p_ent"] = safe(entropy, n_common_arr)
        
        f_common_union_arr = np.array(g_f_common_union[etype])
        f_common_union_mean, f_common_union_var = mean_var_sparse(
            f_common_union_arr, 
            n_edge_pairs
        )
        metrics[f"f_common_union_{etype}_mean"] = f_common_union_mean
        metrics[f"f_common_union_{etype}_var"] = f_common_union_var
        
        f_common_min_arr = np.array(g_f_common_min[etype])
        f_common_min_mean, f_common_min_var = mean_var_sparse(
            f_common_min_arr, 
            n_edge_pairs
        )
        metrics[f"f_common_min_{etype}_mean"] = f_common_min_mean
        metrics[f"f_common_min_{etype}_var"] = f_common_min_var

    return metrics


def compute_metrics(json_fpath):
    conversation = json.load(gzip.open(json_fpath))
    metrics = embeddedness_metrics(conversation, remove_root=False)
    return metrics


#
# EMBEDDEDNESS METRICS COMPUTATION 
#
def compute_embeddedness_metrics(dataset, n_jobs=1, limit=None):

    print("--- Embeddedness Metrics ---")
    print(f"Dataset: {dataset}")
    print(f"Num Jobs: {n_jobs}")
    print(f"Limit: {limit}")
    print("----------------------------")
    
    # paths
    conf = Config(dataset)
    
    output_fpath = f"{conf.data_root}/emb_metrics.csv"

    # iterator
    json_fpaths = json_paths_iter(conf.conversations_jsons_dir, limit=limit)

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
    print("Outputting embeddings metrics to CSV ...")
    write_dicts_to_csv(metrics, output_fpath)

    print("Done!")


@click.command()
@click.option('--dataset', required=True,
    type=click.Choice(["news", "midterms"]))
@click.option('--n_jobs', required=True, type=int)
@click.option('--limit', default=None, type=int)
def main(dataset, n_jobs, limit):
    
    compute_embeddedness_metrics(
        dataset, 
        n_jobs=n_jobs, 
        limit=limit
    )


if __name__ == "__main__":
    main()

# END
