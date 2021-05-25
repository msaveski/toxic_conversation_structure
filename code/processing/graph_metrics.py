
import numpy as np
import networkx as nx

from community import best_partition as louvain
from community import modularity as lv_modularity
from networkx.classes.function import density
from networkx.algorithms.shortest_paths.unweighted import all_pairs_shortest_path_length
from networkx.algorithms.core import core_number, k_core, k_truss
from networkx.algorithms.components import connected_components
from networkx.algorithms.cluster import transitivity, average_clustering
from networkx.algorithms.centrality import betweenness_centrality
from networkx.algorithms.centrality import closeness_centrality
from networkx.algorithms.centrality import eigenvector_centrality_numpy
from networkx.algorithms.link_analysis.pagerank_alg import pagerank_numpy
from networkx.algorithms.community.modularity_max import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity as nx_modularity
from networkx.linalg.algebraicconnectivity import algebraic_connectivity

from _assortativity import numeric_attribute_correlation
from _common import safe, h_index, gini, entropy
from subgraph_metrics import dyadic_census

eps = 1e-16

CC_k_thresholds = [1, 2, 3, 4, 5, 10]
k_core_ks = [1, 2, 3, 4, 5]
k_truss_ks = [1, 2, 3, 4, 5]



def centralization_metrics(G, prefix=""):
    # NB: G can be either directed or undirected network
    # Metrics:
    # (betweennes / closeness / eigenvector / pagerank)
    
    # betweenness 
    # => expensive
    # sample: k=min(10, len(G))
    betweenness = betweenness_centrality(G, normalized=True)
    betweenness_arr = np.fromiter(betweenness.values(), dtype=np.float)
    betweenness_mean = np.mean(np.max(betweenness_arr) - betweenness_arr)
    
    # closeness 
    # => expensive
    # NB: normilizes by the CC size
    closeness = closeness_centrality(G, wf_improved=False)
    closeness_arr = np.fromiter(closeness.values(), dtype=np.float)
    closeness_mean = np.mean(np.max(closeness_arr) - closeness_arr)
    
    # eigenvector
    eigenvec_mean = None
    if len(G) > 2:
        try:
            eigenvec = eigenvector_centrality_numpy(G)
            eigenvec_arr = np.fromiter(eigenvec.values(), dtype=np.float)
            eigenvec_mean = np.mean(np.max(eigenvec_arr) - eigenvec_arr)
        except:
            eigenvec_mean = None
    
    # pagerank
    try:
        pagerank = pagerank_numpy(G)
        pagerank_arr = np.fromiter(pagerank.values(), dtype=np.float)
        pagerank_mean = np.mean(np.max(pagerank_arr) - pagerank_arr)
    except:
        pagerank_mean = None 
    
    centralization = {
        f"cent{prefix}_betweenness_mean": betweenness_mean,
        f"cent{prefix}_closeness_mean": closeness_mean, 
        f"cent{prefix}_eigenvec_mean": eigenvec_mean,
        f"cent{prefix}_pagerank_mean": pagerank_mean
    }    
    
    return centralization


def compute_modularity_metrics(G):    
    assert type(G) is nx.Graph
    
    """
    # Clauset-Newman-Moore greedy modularity maximization
    # NB: much slower than Louvain => SKIP
    if len(G.edges()) > 0:
        greedy_partitions = greedy_modularity_communities(G)
        modularity_gready = nx_modularity(G, greedy_partitions)
    else:
        modularity_gready = None
    """
    
    # Louvain greedy modularity maximization
    modularity_louvain = None
    
    if len(G.edges()) > 0:
        louvain_partitions = louvain(G, random_state=0)
        modularity_louvain = lv_modularity(louvain_partitions, G)
    
    metrics = {
        # "modularity_gready": modularity_gready,
        "modularity_louvain": modularity_louvain
    }
    
    return metrics


def fraction_of_connected_node_pairs(G):
    # computes how many pairs of nodes have a path between each other
    if len(G) < 2:
        return None
    
    pairs_path_lens = all_pairs_shortest_path_length(G)

    n_connected_pairs = 0
    for i, i_p in pairs_path_lens:
        for j, l in i_p.items():
            if l > 0:
                n_connected_pairs += 1

    f_connected_pairs = n_connected_pairs / (len(G) * (len(G) - 1.0))
    return f_connected_pairs


def compute_dyad_metrics(dyad_freq):
    dyad_metrics = {}
    dyad_freq_sum = sum(dyad_freq.values())
    
    for d_type, freq in dyad_freq.items():
        if dyad_freq_sum > 0:
            dyad_metrics[f"dyads_n_e{d_type}"] = freq
            dyad_metrics[f"dyads_f_e{d_type}"] = freq / dyad_freq_sum
        else:
            dyad_metrics[f"dyads_n_e{d_type}"] = None
            dyad_metrics[f"dyads_f_e{d_type}"] = None
            
    return dyad_metrics


#
# Undirected Graph Metrics
#
def compute_undirected_graph_metrics(G):
    assert type(G) is nx.Graph
    
    # degrees stats
    degrees = np.array([i for _, i in G.degree])
    degrees_k_freq = np.unique(degrees, return_counts=True)[1]
    degrees_corr = numeric_attribute_correlation(G, dict(G.degree), dict(G.degree))
    
    # clustering
    global_clustering = transitivity(G)
    local_clustering_mean = average_clustering(G)
    
    # fraction of connected node pairs (any path len)
    f_connected_node_pairs = fraction_of_connected_node_pairs(G)
    
    # centralization
    cent_metrics = centralization_metrics(G, prefix="_ud")
    
    # modularity
    modularity_metrics = compute_modularity_metrics(G)
    
    # largest CC
    CC1_nodes = max(connected_components(G), key=len)
    CC1 = G.subgraph(CC1_nodes).copy()
    f_CC1_nodes = len(CC1) / len(G)
    
    # algebraic_connectivity of the largest CC
    algebraic_connectivity_CC1 = None
    if len(CC1) > 2:
        try:
            algebraic_connectivity_CC1 = algebraic_connectivity(CC1, seed=0)
        except:
            algebraic_connectivity_CC1 = None 
        
        
    # connected components
    CC = connected_components(G)
    CC_sizes = np.array([len(cc_i) for cc_i in CC])
    
    CC_metrics = {}
    for k in CC_k_thresholds:
        CC_metrics[f"n_CC_{k}"] = np.sum(CC_sizes >= k)
        
    # k-core
    k_core_metrics = {}
    G_core_number = core_number(G)
    
    for k in k_core_ks:
        k_core_subgraph = k_core(G, k=k, core_number=G_core_number)
        k_core_metrics[f"core_{k}_n_nodes"] = len(k_core_subgraph.nodes)
        k_core_metrics[f"core_{k}_n_edges"] = len(k_core_subgraph.edges)
        k_core_metrics[f"core_{k}_density"] = density(k_core_subgraph)
        k_core_metrics[f"core_{k}_n_CC"] = len(list(connected_components(k_core_subgraph)))

    # k-truss
    k_truss_metrics = {}
    
    for k in k_truss_ks:
        k_truss_subgraph = k_truss(G, k=k)
        k_truss_metrics[f"truss_{k}_n_nodes"] = len(k_truss_subgraph.nodes)
        k_truss_metrics[f"truss_{k}_n_edges"] = len(k_truss_subgraph.edges)
        k_truss_metrics[f"truss_{k}_density"] = density(k_truss_subgraph)
        k_truss_metrics[f"truss_{k}_n_CC"] = len(list(connected_components(k_truss_subgraph)))    
    
    metrics = {
        "n_edges_ud": len(G.edges()),
        "density_ud": density(G),
        # degree stats
        "degrees_mean": safe(np.mean, degrees),
        "degrees_var": safe(np.var, degrees),
        "degrees_hidx": safe(h_index, degrees),
        "degrees_gini": safe(gini, degrees + eps),
        "degrees_f0": safe(np.mean, (degrees == 0)),
        "degrees_corr": degrees_corr,
        "degrees_pk_ent": entropy(degrees_k_freq),
        "degrees_pk_gini": gini(degrees_k_freq),
        # fraction of connected node pairs with path of any length
        "f_connected_node_pairs_ud": f_connected_node_pairs,
        # clustering coefficients
        "global_clustering_ud": global_clustering,
        "local_clustering_mean_ud": local_clustering_mean,
        # centralization
        **cent_metrics,
        # modularity
        **modularity_metrics,
        # fraction of nodes in the largest CC
        "f_CC1_nodes": f_CC1_nodes,
        # algebraic connectivity of the largest CC
        "algebraic_connectivity_CC1": algebraic_connectivity_CC1,
        # connected components
        **CC_metrics,
        # k-core 
        **k_core_metrics,
        # k-truss
        **k_truss_metrics
    }
    
    return metrics


#
# Directed Graph Metrics
#
def compute_directed_graph_metrics(G):
    assert type(G) is nx.DiGraph
    
    n_edges = len(G.edges)
    
    # in & out degree stats
    in_degrees = np.array([n for _, n in G.in_degree()])
    out_degrees = np.array([n for _, n in G.out_degree()])

    in_degrees_k_freq = np.unique(in_degrees, return_counts=True)[1]
    out_degrees_k_freq = np.unique(out_degrees, return_counts=True)[1]
    
    out_in_degrees_corr = numeric_attribute_correlation(
        G, 
        dict(G.out_degree), 
        dict(G.in_degree)
    )

    # dyad metrics
    dyad_freq = dyadic_census(G)
    dyad_metrics = compute_dyad_metrics(dyad_freq)
    
    # reciprocity
    reciprocity = None
    if n_edges > 0:
        # based on networkx definition
        reciprocity = 2 * dyad_freq["2"] / (dyad_freq["1"] + 2 * dyad_freq["2"])
        
    # clustering
    global_clustering = transitivity(G)
    local_clustering_mean = average_clustering(G)
    
    # fraction of connected node pairs (any path len)
    f_connected_node_pairs = fraction_of_connected_node_pairs(G)
    
    # centralization
    cent_metrics = centralization_metrics(G, prefix="_di")
    
    metrics = {
        "n_edges_di": len(G.edges),
        "density_di": density(G),
        "reciprocity": reciprocity,
        # in_degree
        "in_degrees_mean": safe(np.mean, in_degrees),
        "in_degrees_var": safe(np.var, in_degrees),
        "in_degrees_hidx": safe(h_index, in_degrees),
        "in_degrees_gini": safe(gini, in_degrees + eps),
        "in_degrees_f0": safe(np.mean, (in_degrees == 0)),        
        "in_degrees_pk_ent": entropy(in_degrees_k_freq),
        "in_degrees_pk_gini": gini(in_degrees_k_freq),
        # out_degree
        "out_degrees_mean": safe(np.mean, out_degrees),
        "out_degrees_var": safe(np.var, out_degrees),
        "out_degrees_hidx": safe(h_index, out_degrees),
        "out_degrees_gini": safe(gini, out_degrees + eps),
        "out_degrees_f0": safe(np.mean, (out_degrees == 0)),
        "out_degrees_pk_ent": entropy(out_degrees_k_freq),
        "out_degrees_pk_gini": gini(out_degrees_k_freq),
        # degree assortativity
        "out_in_degrees_corr": out_in_degrees_corr,
        # dyad metric
        **dyad_metrics,
        # fraction of connected node pairs with path of any length
        "f_connected_node_pairs_di": f_connected_node_pairs,
        # clustering coefficients
        "global_clustering_di": global_clustering,
        "local_clustering_mean_di": local_clustering_mean,
        # centralization
        **cent_metrics
    }
    
    return metrics


#
# Twitter num of friends and num of followers metrics
#
def n_friends_followers_stats(G, net_features, root_user_id):
    n_friends, n_followers = [], []
    user_n_friends, user_n_followers = {}, {}
    
    triples = zip(
        net_features["user_ids"],
        net_features["n_friends"],
        net_features["n_followers"]
    )
    for u_id, u_n_friends, u_n_followers in triples:
        if root_user_id is not None and u_id == root_user_id:
            continue
            
        if u_n_friends is not None:
            n_friends.append(u_n_friends)
            
        if u_n_followers is not None:
            n_followers.append(u_n_followers)
        
        # Nones are ok, handled by the assortativity method
        user_n_friends[u_id] = u_n_friends
        user_n_followers[u_id] = u_n_followers
    
    n_friends = np.array(n_friends)
    n_followers = np.array(n_followers)
    
    # compute assortitivity
    n_friends_n_friends_corr = numeric_attribute_correlation(G, user_n_friends, user_n_friends)
    n_friends_n_followers_corr = numeric_attribute_correlation(G, user_n_friends, user_n_followers)
    n_followers_n_friends_corr = numeric_attribute_correlation(G, user_n_followers, user_n_friends)
    n_followers_n_followers_corr = numeric_attribute_correlation(G, user_n_followers, user_n_followers)    
    
    metrics = {
        # n_friends stats
        "user_n_friends_mean": safe(np.mean, n_friends),
        "user_n_friends_var": safe(np.var, n_friends),
        "user_n_friends_hidx": safe(h_index, n_friends),
        "user_n_friends_gini": safe(gini, n_friends + eps),
        # n_followers stats
        "user_n_followers_mean": safe(np.mean, n_followers),
        "user_n_followers_var": safe(np.var, n_followers),
        "user_n_followers_hidx": safe(h_index, n_followers),
        "user_n_followers_gini": safe(gini, n_followers + eps),
        # assortativity
        "n_friends_n_friends_corr": n_friends_n_friends_corr, 
        "n_friends_n_followers_corr": n_friends_n_followers_corr,
        "n_followers_n_friends_corr": n_followers_n_friends_corr,
        "n_followers_n_followers_corr": n_followers_n_followers_corr
    }
    
    return metrics

# END