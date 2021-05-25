
import csv
import gzip
import itertools
import click
import ujson as json

from tqdm import tqdm
from joblib import Parallel, delayed

from _config import Config
from utils import json_paths_iter, is_toxic
from tree_conversions import tree_json_to_tree_lib



def get_tree_edges_tuples(tree, skip_root=True, splits_only=False):
    edge_tuples = []

    for parent in tree.all_nodes_itr():
        if skip_root and parent.is_root():
            continue
                        
        parent_tweet_id = parent.identifier
        parent_user_id = parent.data["user_id"]
        
        if splits_only and len(tree.children(parent.identifier)) < 2:
            continue        

        for child in tree.children(parent_tweet_id):
            child_tweet_id = child.identifier
            child_user_id = child.data["user_id"]
            
            if parent_user_id == child_user_id:
                continue

            edge_tuples.append(
                (
                    (parent_tweet_id, parent_user_id),
                    (child_tweet_id, child_user_id)
                )
            )

    return edge_tuples


def preprocess_network_data(conversation_json):

    net_features = conversation_json["network_features"]

    # set missing user ids
    missing_ids = set(net_features["missing_user_ids"])

    # user id -> user index
    user_id_to_idx = {u_id: u_idx \
        for u_idx, u_id in enumerate(net_features["user_ids"])}

    # edges: set of user id tuples
    edges_user_indexes = set([(i, j) for i, j in net_features["network"]])

    # embeddings: (user_idx, user_idx) -> emb (number of common friends)
    emb_user_indexes = {(i, j): emb \
        for i, j, emb in net_features["network_intersections"]}

    return {
        "missing_ids": missing_ids,
        "user_id_to_idx": user_id_to_idx,
        "edges_user_indexes": edges_user_indexes,
        "emb_user_indexes": emb_user_indexes,
        "n_friends": net_features["n_friends"],
        "n_followers": net_features["n_followers"]
    }


def get_dyad_type(parent_user_idx, child_user_idx, net_data):
    # missing parent/child network data
    if parent_user_idx is None or child_user_idx is None:
        return None

    edges = net_data["edges_user_indexes"]

    parent_child_edge = (parent_user_idx, child_user_idx) in edges
    child_parent_edge = (child_user_idx, parent_user_idx) in edges

    if parent_child_edge == True and child_parent_edge == True:
        return "O==O"

    if parent_child_edge == True and child_parent_edge == False:
        return "O->O"

    if parent_child_edge == False and child_parent_edge == True:
        return "O<-O"

    if parent_child_edge == False and child_parent_edge == False:
        return "O  O"

    raise Exception("Invalid parent_tox or child_tox values")


def get_dyad_embeddness(parent_user_idx, child_user_idx, net_data):
    # missing parent/child network data
    if parent_user_idx is None or child_user_idx is None:
        return None

    emb_user_indexes = net_data["emb_user_indexes"]

    # symmetric: i > j by construction
    parent_child_tuple = tuple(sorted((parent_user_idx, child_user_idx)))

    if parent_child_tuple not in emb_user_indexes:
        return 0

    return emb_user_indexes[parent_child_tuple]


def process_conversation(json_fpath, tox_threshold, splits_only, skip_root):
    dyad_tuples = []

    conversation = json.load(gzip.open(json_fpath))    

    root_tweet_id = conversation["reply_tree"]["tweet"]    
    
    # parse tree
    tree = tree_json_to_tree_lib(conversation)
    
    net_data = preprocess_network_data(conversation)

    toxicity_scores = conversation["toxicity_scores"]

    # get edge tuples
    edge_tuples = get_tree_edges_tuples(
        tree, 
        skip_root=skip_root, 
        splits_only=splits_only
    )
    
    if len(edge_tuples) == 0:
        return []

    for parent_info, child_info in edge_tuples:
        # parent
        parent_tweet_id, parent_user_id = parent_info
        parent_tox = is_toxic(toxicity_scores, parent_tweet_id, tox_threshold)
        parent_user_idx = net_data["user_id_to_idx"].get(parent_user_id, None)

        if parent_user_idx is not None:
            parent_n_friends = net_data["n_friends"][parent_user_idx]
            parent_n_followers = net_data["n_followers"][parent_user_idx]
        else:
            parent_n_friends = None
            parent_n_followers = None

        # child
        child_tweet_id, child_user_id = child_info
        child_tox = is_toxic(toxicity_scores, child_tweet_id, tox_threshold)
        child_user_idx = net_data["user_id_to_idx"].get(child_user_id, None)

        if child_user_idx is not None:
            child_n_friends = net_data["n_friends"][child_user_idx]
            child_n_followers = net_data["n_followers"][child_user_idx]
        else:
            child_n_friends = None
            child_n_followers = None

        # sanity checks
        if parent_user_idx is None:
            assert parent_user_id in net_data["missing_ids"]

        if child_user_idx is None:
            assert child_user_id in net_data["missing_ids"]

        # edge type
        dyad_type = get_dyad_type(parent_user_idx, child_user_idx, net_data)

        # embeddings (i.e., number of common friends x followers)
        dyad_n_common_friends = get_dyad_embeddness(
            parent_user_idx, 
            child_user_idx, 
            net_data
        )

        # create tuple
        dyad_tuple = (
            root_tweet_id,
            parent_tox,
            parent_n_friends,
            parent_n_followers,
            child_tox,
            child_n_friends,
            child_n_followers,
            dyad_type,
            dyad_n_common_friends
        )
        
        dyad_tuples.append(dyad_tuple)
            
    return dyad_tuples


#
# DYAD METRICS COMPUTATION
#
def compute_dyad_metrics(dataset, n_jobs, limit):

    # hard-coding some settings
    toxicity_threshold = 0.531
    splits_only = False
    skip_root = True
    
    print("--- Dyad Metrics ---")
    print(f"Dataset: {dataset}")
    print(f"Toxicity threshold: {toxicity_threshold}")
    print(f"Num Jobs: {n_jobs}")
    print(f"Limit: {limit}")
    print("----------------------------")

    conf = Config(dataset)

    output_fpath = f"{conf.data_root}/dyad_metrics.csv"

    json_fpaths = json_paths_iter(conf.conversations_jsons_dir, limit=limit)

    # compute metrics
    print("Computing metrics ...")

    if n_jobs == 1:
        metrics = [
            process_conversation(
                json_fpath, 
                toxicity_threshold, 
                splits_only, 
                skip_root
            )
            for json_fpath in tqdm(json_fpaths)
        ]
    else:
        parallel = Parallel(n_jobs=n_jobs, verbose=10)
        metrics = parallel(
            delayed(process_conversation)(
                    json_fpath, 
                    toxicity_threshold, 
                    splits_only, 
                    skip_root
                ) \
                for json_fpath in json_fpaths
            )

    # flatten the results
    metrics = list(itertools.chain.from_iterable(metrics))
    print(len(metrics))
    
    # output to CSV
    fields = [
        "root_tweet_id",
        "parent_tox",
        "parent_n_friends",
        "parent_n_followers",
        "child_tox",
        "child_n_friends",
        "child_n_followers",
        "dyad_type",
        "dyad_n_common_friends"
    ]

    with open(output_fpath, "w") as fout:
        writer = csv.writer(fout)
        writer.writerow(fields)
        writer.writerows(metrics)

    print("Done!")


@click.command()
@click.option('--dataset', required=True,
    type=click.Choice(["news", "midterms"]))
@click.option('--n_jobs', required=True, type=int)
@click.option('--limit', default=None, type=int)
def main(dataset, n_jobs, limit):
    
    compute_dyad_metrics(
        dataset, 
        n_jobs, 
        limit
    )


if __name__ == "__main__":
    main()

# END
