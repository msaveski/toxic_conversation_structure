
import gzip
import click
import ujson as json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from joblib import Parallel, delayed

from _config import Config
from utils import json_paths_iter, write_dicts_to_csv
from tree_conversions import tree_json_to_tree_lib
from _common import h_index, entropy, gini



def wiener_index(T):
    """
    Taken from:
        Goel, Sharad, et al.
        "The structural virality of online diffusion."
        Management Science (2015).
        Appendix A

    Analysis of the lower and upper bound of the winner index on trees:
        Schmuck, Nina Sabine. The Wiener index of a graph. Thesis, 2010.
    """
    size, sum_sizes, sum_sizes_sq = subtree_moments(T, T.root)
    avg_dist = (2.0 * size / (size - 1)) * (sum_sizes / size - sum_sizes_sq / size**2)
    return avg_dist


def subtree_moments(T, root_node):

    if len(T.children(root_node)) == 0:
        # base case
        size = 1
        sum_sizes = 1
        sum_sizes_sq = 1

    else:
        # recurse over the children of the root
        size = 0
        sum_sizes = 0
        sum_sizes_sq = 0

        for child in T.children(root_node):
            sub_tree_root = child.identifier
            c_size, c_sum_sizes, c_sum_sizes_sq = subtree_moments(T, sub_tree_root)

            size += c_size
            sum_sizes += c_sum_sizes
            sum_sizes_sq += c_sum_sizes_sq

        # account for the root node
        size += 1
        sum_sizes += size
        sum_sizes_sq += size**2

    return size, sum_sizes, sum_sizes_sq


def tree_metrics(conversation, depth_n_levels):

    reply_tree = tree_json_to_tree_lib(conversation)

    # root info
    root_tweet_id = conversation["reply_tree"]["tweet"]

    # init
    n_tweets = 0

    depth_n_nodes = defaultdict(int)
    depth_1_nodes_with_reply = 0
    depth_1_nodes_subtree_sizes = []
    depth_1_nodes_subtree_depth_size_ratio = []

    user_n_tweets = defaultdict(int)

    n_children = []

    nodes_depths = []

    leaves_depths = []


    for node in reply_tree.all_nodes():
        
        tweet_id = node.identifier
        user_id = node.data["user_id"]
        
        node_depth = reply_tree.level(tweet_id)
        node_n_children = len(reply_tree.children(tweet_id))
        
        # skip root tweet
        if node_depth == 0:
            continue
        
        depth_n_nodes[node_depth] += 1
        
        if node_depth == 1:
            subtree = reply_tree.subtree(tweet_id)
            subtree_size = len(subtree)
            subtree_depth_size = subtree.depth() /subtree_size

            depth_1_nodes_subtree_sizes.append(subtree_size)
            depth_1_nodes_subtree_depth_size_ratio.append(subtree_depth_size)
            depth_1_nodes_with_reply += int(node_n_children > 0)
        
        user_n_tweets[user_id] += 1
        
        n_children.append(node_n_children)
        
        nodes_depths.append(node_depth)
        
        if node.is_leaf():
            leaves_depths.append(node_depth)
            
        n_tweets += 1

    # 
    # combining all statistics
    #
    n_users = len(user_n_tweets)
        
    # depth: the depth of the deepest node
    depth = reply_tree.depth()

    # width: max breadth at any depth
    width = max(depth_n_nodes.values())

    # wiener_index aka structural virality
    w_ind = wiener_index(reply_tree)

    depth_n_nodes_ratio = depth / n_tweets

    lvl1_replies_frac = depth_n_nodes[1] / n_tweets
    lvl1_replies_with_replies_frac = depth_1_nodes_with_reply / depth_n_nodes[1]
    lvl1_max_subtree_depth_size_ratio = max(depth_1_nodes_subtree_depth_size_ratio)

    # make vars numpy arrays
    user_n_tweets_arr = np.array(list(user_n_tweets.values()))
    n_children_arr = np.array(n_children)
    depth_n_nodes_arr = np.array(list(depth_n_nodes.values()))
    depth_1_nodes_subtree_sizes_arr = np.array(depth_1_nodes_subtree_sizes)
    nodes_depths_arr = np.array(nodes_depths)
    leaves_depths_arr = np.array(leaves_depths)


    metrics = {
        "root_tweet_id": root_tweet_id,
        "n_tweets": n_tweets,
        "n_users": n_users,
        "depth": depth,
        "width": width,
        "wiener_index": w_ind,
        "depth_n_nodes_ratio": depth_n_nodes_ratio,
        "lvl1_replies_frac": lvl1_replies_frac,
        "lvl1_replies_with_replies_frac": lvl1_replies_with_replies_frac,
        "lvl1_subtree_sizes_ent": entropy(depth_1_nodes_subtree_sizes_arr),
        "lvl1_subtree_sizes_gini": gini(depth_1_nodes_subtree_sizes_arr),
        "lvl1_max_subtree_depth_size_ratio": lvl1_max_subtree_depth_size_ratio,
        # number of tweets per user
        "user_n_tweets_mean": np.mean(user_n_tweets_arr),
        "user_n_tweets_var": np.var(user_n_tweets_arr),
        "user_n_tweets_hidx": h_index(user_n_tweets_arr),
        "user_n_tweets_ent": entropy(user_n_tweets_arr),
        "user_n_tweets_gini": gini(user_n_tweets_arr),
        # number of children per node 
        # NB: mean => branching factor
        "n_children_mean": np.mean(n_children_arr),
        "n_children_var": np.var(n_children_arr),
        "n_children_hidx": h_index(n_children_arr),
        # number of nodes at each depth / level
        "depth_n_nodes_mean": np.mean(depth_n_nodes_arr),
        "depth_n_nodes_var": np.var(depth_n_nodes_arr),
        "depth_n_nodes_hidx": h_index(depth_n_nodes_arr),
        "depth_n_nodes_ent": entropy(depth_n_nodes_arr),
        "depth_n_nodes_gini": gini(depth_n_nodes_arr),
        # depth values of all nodes
        "nodes_depths_mean": np.mean(nodes_depths_arr),
        "nodes_depths_var": np.var(nodes_depths_arr),
        "nodes_depths_hidx": h_index(nodes_depths_arr),
        "nodes_depths_ent": entropy(nodes_depths_arr),
        "nodes_depths_gini": gini(nodes_depths_arr),
        # depth value of the leaves
        "leaves_depths_mean": np.mean(leaves_depths_arr),
        "leaves_depths_var": np.var(leaves_depths_arr),
        "leaves_depths_hidx": h_index(leaves_depths_arr),
        "leaves_depths_ent": entropy(leaves_depths_arr),
        "leaves_depths_gini": gini(leaves_depths_arr),   
    }

    # depth metrics
    for d in range(1, depth_n_levels + 1):
        metrics[f"d_{d}_n_nodes"] = depth_n_nodes[d] if d <= depth else None

    # add to the list of all metrics
    return metrics


def compute_metrics(json_fpath, depth_n_levels):
    conversation = json.load(gzip.open(json_fpath))
    metrics = tree_metrics(conversation, depth_n_levels)
    return metrics


#
# TREE METRICS COMPUTATION
#
def compute_tree_metrics(dataset, depth_n_levels, n_jobs, limit):

    print("--- Reply Tree Metrics ---")
    print(f"Dataset: {dataset}")
    print(f"Depth metrics, number of levels: {depth_n_levels}")
    print(f"Num Jobs: {n_jobs}")
    print(f"Limit: {limit}")
    print("---------------------------")

    # paths
    conf = Config(dataset)

    output_fpath = f"{conf.data_root}/tree_metrics.csv"

    json_fpaths = json_paths_iter(
        conf.conversations_no_embs_jsons_dir, 
        limit=limit
    )

    # compute metrics
    print("Computing metrics ...")

    if n_jobs == 1:
        metrics = [compute_metrics(json_fpath, depth_n_levels) \
            for json_fpath in tqdm(json_fpaths)]
    else:
        parallel = Parallel(n_jobs=n_jobs, verbose=10)
        metrics = parallel(
            delayed(compute_metrics)(json_fpath, depth_n_levels) \
                for json_fpath in json_fpaths
            )            

    print("Output:", len(metrics))

    # output to csv
    print("Outputting tree metrics to CSV ...")
    write_dicts_to_csv(metrics, output_fpath)

    print("Done!")


@click.command()
@click.option('--dataset', required=True, 
    type=click.Choice(["news", "midterms"], case_sensitive=False))
@click.option("--depth_n_levels", default=15)
@click.option('--n_jobs', required=True, type=int)
@click.option('--limit', default=None, type=int)
def main(dataset, depth_n_levels, n_jobs, limit):

    # python tree_metrics.py --dataset=news --n_jobs=1 --limit=10

    compute_tree_metrics(
        dataset, 
        depth_n_levels, 
        n_jobs, 
        limit
    )
    

if __name__ == "__main__":
    main()

# END
