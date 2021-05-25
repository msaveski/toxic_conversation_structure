
import gzip
import click
import numpy as np
import ujson as json

from tqdm import tqdm
from joblib import Parallel, delayed
from community import modularity

from _config import Config
from _assortativity import numeric_attribute_correlation, safe_pearsonr
from _assortativity import categorical_attribute_correlation
from _assortativity import categorical_attribute_correlation_iter
from utils import json_paths_iter, write_dicts_to_csv
from tree_conversions import get_tree_user_edges, tree_to_nx_user_graph
from graph_conversions import follow_json_to_nx_graph



def entropy(n_left, n_right):
    """
    Left/right entropy.
    """
    if n_left == 0 or n_right == 0:
        return 0

    n = n_left + n_right

    pl = n_left / n
    pr = n_right / n

    h = -(pl * np.log2(pl) + pr * np.log2(pr))

    return float(h)


def alignment_summary_stats(scores, prefix=None):
    """
    Compute summary stats over the alignments for the set of users
    who participated in the conversation.
    """

    fields = [
        "n", "mean", "std",
        "q0", "q25", "q50", "q75", "q100", "iqr",
        "n_left", "n_right", "entropy"
    ]

    # no scores => return a dictionary of Nones
    if len(scores) == 0 and prefix is None:
        return {field: None for field in fields}

    if len(scores) == 0 and prefix is not None:
        return {f"{prefix}_{field}": None for field in fields}

    # compute stats
    scores = np.array(scores)

    mean = float(np.mean(scores))

    std = float(np.std(scores))

    quartiles = (np
        .quantile(scores, [0, 0.25, 0.5, 0.75, 1])
        .tolist())

    quartiles = dict(zip(('q0', 'q25', 'q50', 'q75', 'q100'), quartiles))

    IQR = quartiles["q75"] - quartiles["q25"]

    n_left = int(np.sum(scores <= 0))
    n_right = int(np.sum(scores > 0))

    h = entropy(n_left, n_right)

    result = {
        "n": len(scores),
        "mean": mean,
        "std": std,
        "q0": quartiles["q0"],
        "q25": quartiles["q25"],
        "q50": quartiles["q50"],
        "q75": quartiles["q75"],
        "q100": quartiles["q100"],
        "iqr": IQR,
        "n_left": n_left,
        "n_right": n_right,
        "entropy": h
    }

    # add filed prefix
    if prefix:
        result = {f"{prefix}_{key}": val for key, val in result.items()}

    return result
    

def graph_corr_metrics(G, u_alg_cat, u_alg_num, u_alg_groups, prefix):
    
    # make a subgraph view of the node that have alignments
    # NB: it skips nodes not present in G (may happen when the user friend 
    #     list is missing)
    Gs = G.subgraph(u_alg_cat.keys())
    
    # assortitivity
    corr_cat = categorical_attribute_correlation(
        G, 
        u_alg_cat, 
        u_alg_cat, 
        categories = ["L", "R"]
    )
    corr_num = numeric_attribute_correlation(G, u_alg_num, u_alg_num)
    
    metrics = {
        f"{prefix}_n_nodes": len(Gs.nodes()),
        f"{prefix}_alg_cat_corr": corr_cat,
        f"{prefix}_alg_num_corr": corr_num
    }
    
    # modularity (undirected graphs only)
    if not Gs.is_directed():
        Gs_mod = None
        if Gs.size() > 0:
            Gs_mod = modularity(u_alg_groups, Gs)
        metrics[f"{prefix}_alg_modularity"] = Gs_mod
        
    return metrics     


def tree_corr_metrics(conversation, u_alg_cat, u_alg_num):
    
    reply_edges = get_tree_user_edges(
        conversation["reply_tree"], 
        conversation["tweets"]
    )
    
    cat_pairs = []
    num_attr1 = []
    num_attr2 = []
    
    for u_i, u_j in reply_edges:    
        if u_i not in u_alg_num or u_j not in u_alg_num:
            continue
        cat_pairs.append((u_alg_cat[u_i], u_alg_cat[u_j]))
        num_attr1.append(u_alg_num[u_i])
        num_attr2.append(u_alg_num[u_j])
    
    cat_corr = categorical_attribute_correlation_iter(
        cat_pairs, 
        categories=["L", "R"]
    )
    num_corr = safe_pearsonr(num_attr1, num_attr2)
    
    metrics = {
        "tree_alg_cat_corr": cat_corr,
        "tree_alg_num_corr": num_corr
    }
    
    return metrics


def polarization_metrics(conversation, remove_root):
    # fetch user alignment
    user_alignments = conversation["alignment_scores"]
    # user_alignments = {u_id: alg for u_id, alg in user_alignments.items()}

    # fetch root info
    root_tweet_id = conversation["reply_tree"]["tweet"]
    root_user_id = conversation["tweets"][root_tweet_id]["user_id"]

    # user_ids of those who are in the follow network
    net_user_ids = set(conversation["network_features"]["user_ids"])
        
    # compile all user ids
    tweets = conversation["tweets"].values()
    user_ids = {tweet["user_id"] for tweet in tweets}
    
    if remove_root and root_user_id in user_ids:
        user_ids.remove(root_user_id)
    
    # compile user alignments (if available)
    u_alg_cat = {}
    u_alg_num = {}
    u_alg_groups_reply = {}
    u_alg_groups_follow = {}
    
    for u_id in user_ids:
        if u_id not in user_alignments:
            continue

        alg_num = user_alignments[u_id]
        alg_cat = "L" if alg_num < 0 else "R"
        
        u_alg_num[u_id] = alg_num
        u_alg_cat[u_id] = alg_cat
        u_alg_groups_reply[u_id] = alg_cat
        
        if u_id in net_user_ids:
            u_alg_groups_follow[u_id] = alg_cat
    
    # alignment summary statistics
    alg_num_values = list(u_alg_num.values())
    alg_metrics = alignment_summary_stats(alg_num_values, prefix="alg")
    
    # reply tree metrics
    # NB: this may count the same (replier, poster) pair multiple times
    reply_tree_metrics = tree_corr_metrics(conversation, u_alg_cat, u_alg_num)
    
    # follow/reply graph metrics
    G_di_follow = follow_json_to_nx_graph(
        conversation, 
        directed=True, 
        remove_root=remove_root
    )
    G_ud_follow = follow_json_to_nx_graph(
        conversation, 
        directed=False, 
        remove_root=remove_root
    )

    G_di_reply = tree_to_nx_user_graph(
        conversation, 
        directed=True, 
        remove_root=remove_root
    )
    G_ud_reply = tree_to_nx_user_graph(
        conversation, 
        directed=False, 
        remove_root=remove_root
    )    
    
    follow_di_metrics = graph_corr_metrics(
        G_di_follow, 
        u_alg_cat, 
        u_alg_num, 
        u_alg_groups_follow, 
        prefix="follow_di"
    )
    
    follow_ud_metrics = graph_corr_metrics(
        G_ud_follow, 
        u_alg_cat, 
        u_alg_num, 
        u_alg_groups_follow, 
        prefix="follow_ud"
    )

    reply_di_metrics = graph_corr_metrics(
        G_di_reply, 
        u_alg_cat, 
        u_alg_num, 
        u_alg_groups_reply, 
        prefix="reply_di"
    )

    reply_ud_metrics = graph_corr_metrics(
        G_ud_reply, 
        u_alg_cat, 
        u_alg_num, 
        u_alg_groups_reply, 
        prefix="reply_ud"
    )
    
    metrics = {
        "root_tweet_id": root_tweet_id,
        **alg_metrics,
        **reply_tree_metrics, 
        **follow_di_metrics,
        **follow_ud_metrics,
        **reply_di_metrics,
        **reply_ud_metrics
    }

    return metrics
    

def compute_metrics(json_fpath):
    conversation = json.load(gzip.open(json_fpath))
    metrics = polarization_metrics(conversation, remove_root=False)
    return metrics    


def compute_polarization_metrics(dataset, n_jobs, limit):
    print("--- Polarization Metrics ---")
    print(f"Dataset: {dataset}")
    print(f"Num Jobs: {n_jobs}")
    print(f"Limit: {limit}")
    print("----------------------------")

    conf = Config(dataset)

    # iterator
    json_fpaths = json_paths_iter(
        conf.conversations_no_embs_jsons_dir, 
        limit=limit
    )

    output_fpath = f"{conf.data_root}/polarization.csv"

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
    print("Outputting polarization metrics to CSV ...")
    write_dicts_to_csv(metrics, output_fpath)    


@click.command()
@click.option('--dataset', required=True, 
    type=click.Choice(["news", "midterms"], case_sensitive=False))
@click.option('--n_jobs', required=True, type=int)
@click.option('--limit', default=None, type=int)
def main(dataset, n_jobs, limit):
    
    compute_polarization_metrics(
        dataset, 
        n_jobs, 
        limit
    )


if __name__ == "__main__":
    main()

# END
