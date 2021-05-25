
import gzip
import ujson as json
import click
import networkx
from tqdm import tqdm
from joblib import Parallel, delayed

from _config import Config
from utils import json_paths_iter, write_dicts_to_csv
from tree_conversions import tree_to_nx_user_graph
from graph_conversions import follow_json_to_nx_graph


def dyadic_census(G):
    # Running time: O(m)

    assert type(G) is networkx.DiGraph

    n = len(G.nodes())
    n_1way_edges = 0
    n_2way_edges_x2 = 0

    for u, v in G.edges():

        if G.has_edge(v, u):
            n_2way_edges_x2 += 1

        else:
            n_1way_edges += 1

    n_2way_edges = int(n_2way_edges_x2 / 2)

    n_pairs = int(n * (n - 1) / 2)

    n_no_edges = n_pairs - (n_1way_edges + n_2way_edges)

    result = {
        "0": n_no_edges,
        "1": n_1way_edges,
        "2": n_2way_edges
    }

    return result


def get_dyad_triad_census(G, prefix=""):

    dyads_census = dyadic_census(G)
    dyads_census = {f"{prefix}_D_{k}": v for k, v in dyads_census.items()}

    triads_census = networkx.triadic_census(G)
    triads_census = {f"{prefix}_T_{k}": v for k, v in triads_census.items()}

    census = {**dyads_census, **triads_census}

    return census


def intersect_graphs(follow_graph, reply_graph):
    """
    Create a copy of the follow graph, but only keep edges where:
    (i replied to j) or (j replied to i), i.e. there is an edge in
    the reply graph between these two nodes in either direction.
    """

    G_intersection = networkx.DiGraph()

    # keep all nodes from the original follow graph
    G_intersection.add_nodes_from(follow_graph.nodes())

    for u, v in follow_graph.edges():
        if not reply_graph.has_edge(u, v) and not reply_graph.has_edge(v, u):
            continue

        G_intersection.add_edge(u, v)

    return G_intersection


def get_conversation_census(conversation_json, remove_root=False):

    root_tweet_id = conversation_json["reply_tree"]["tweet"]

    # reply user graph
    reply_graph = tree_to_nx_user_graph(
        conversation_json,
        directed = True,
        remove_root = remove_root
    )

    reply_graph_census = get_dyad_triad_census(reply_graph, prefix="r")

    # follow user graph
    follow_graph = follow_json_to_nx_graph(
        conversation_json,
        directed = True,
        remove_root = remove_root
    )

    follow_graph_census = get_dyad_triad_census(follow_graph, prefix="f")

    # follow | reply graph
    follow_reply_graph = intersect_graphs(follow_graph, reply_graph)

    follow_reply_graph_census = get_dyad_triad_census(
        follow_reply_graph,
        prefix="fr"
    )

    # sanity checks
    root_user_id = conversation_json["tweets"][root_tweet_id]["user_id"]
    nodes_missing = conversation_json["network_features"]["missing_user_ids"]
    root_missing = int(root_user_id in nodes_missing and remove_root)

    assert len(reply_graph) == (len(follow_graph) + len(nodes_missing) - root_missing)
    assert len(follow_graph) == len(follow_reply_graph)

    return {
        "root_tweet_id": root_tweet_id,
        "f_n_nodes": len(follow_graph),
        **reply_graph_census,
        "r_n_nodes": len(reply_graph),
        **follow_graph_census,
        "fr_n_nodes": len(follow_reply_graph),
        **follow_reply_graph_census
    }


def compute_metrics(json_fpath):
    conversation = json.load(gzip.open(json_fpath))
    metrics = get_conversation_census(conversation)
    return metrics


def compute_subgraph_metrics(dataset, n_jobs, limit):
    print("--- Subgraph Metrics ---")
    print(f"Dataset: {dataset}")
    print(f"Num Jobs: {n_jobs}")
    print(f"Limit: {limit}")
    print("---------------------------")

    # paths
    conf = Config(dataset)

    output_fpath = f"{conf.data_root}/subgraph_metrics.csv"

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
    print("Outputting tree metrics to CSV ...")
    write_dicts_to_csv(metrics, output_fpath)

    print("Done!")


@click.command()
@click.option('--dataset', required=True, 
    type=click.Choice(["news", "midterms"], case_sensitive=False))
@click.option('--n_jobs', required=True, type=int)
@click.option('--limit', default=None, type=int)
def main(dataset, n_jobs, limit):

    compute_subgraph_metrics(
        dataset, 
        n_jobs, 
        limit
    )


if __name__ == "__main__":
    main()

# END
