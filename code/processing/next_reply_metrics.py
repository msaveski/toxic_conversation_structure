
import gzip
import ujson as json
import itertools
import click
import treelib
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from joblib import Parallel, delayed

from _config import Config
from utils import write_dicts_to_csv, is_toxic
from tree_conversions import get_tree_tweet_edges, tree_to_nx_user_graph
from graph_conversions import follow_json_to_nx_graph
from _common import safe, safe_div, entropy, gini

from community import best_partition as louvain
from networkx.algorithms.components import connected_components
from networkx.algorithms.centrality import degree_centrality
from networkx.algorithms.centrality import in_degree_centrality
from networkx.algorithms.centrality import out_degree_centrality
from networkx.algorithms.centrality import closeness_centrality
from networkx.algorithms.centrality import betweenness_centrality
from networkx.algorithms.centrality import eigenvector_centrality_numpy
from networkx.algorithms.link_analysis.pagerank_alg import pagerank_numpy



def tree_stats(tweet_id, tree):
    n_siblings = len(tree.siblings(tweet_id))
    node_depth = tree.depth(tweet_id)

    # find subtree
    subtree = None
    for ancestor_id in tree.rsearch(tweet_id):
        if tree.depth(ancestor_id) == 1:
            subtree = tree.subtree(ancestor_id)
            break        

    subtree_size = subtree.size()
    subtree_depth_size = node_depth / subtree_size
    # size, not counting the root
    subtree_f_size = subtree.size() / (tree.size() - 1)

    stats = {
        "tree_tweet_n_siblings": n_siblings,
        "tree_tweet_depth": node_depth,
        "tree_subtree_size": subtree_size,
        "tree_subtree_depth_size": subtree_depth_size,
        "tree_subtree_f_size": subtree_f_size
    }
    
    return stats


def compute_centralities(G):
    # NB: all  centralities are normalized
    centralities = []

    # degrees
    if G.is_directed():
        centralities.append(("deg_in", in_degree_centrality(G)))
        centralities.append(("deg_out", out_degree_centrality(G)))
    else:
        centralities.append(("deg", degree_centrality(G)))

    # closeness
    CC = closeness_centrality(G, wf_improved=True)
    centralities.append(("closeness", CC))

    # betweenness
    BC = betweenness_centrality(G, normalized=True, seed=0)
    centralities.append(("betweenness", BC))

    # eigenvector centrality
    try:
        EVC = eigenvector_centrality_numpy(G)
        centralities.append(("evc", EVC))
    except:
        pass

    # pagerank
    try:
        PR = pagerank_numpy(G)
        centralities.append(("pagerank", PR))
    except:
        pass
    
    return centralities


def compute_connected_components(SG):
    SG_CC = connected_components(SG)
    CC_map = {}    # node_id => CC_n
    CC_sizes = {}  # CC_n => CC_size

    for CC_n, CC_nodes in enumerate(SG_CC):
        for CC_node in CC_nodes:
            CC_map[CC_node] = CC_n
            CC_sizes[CC_n] = CC_sizes.get(CC_n, 0) + 1

    return {"CC_map": CC_map, "CC_sizes": CC_sizes}


def compute_louvain_partitions(SG):
    LP = louvain(SG, random_state=0)
    LP_sizes = {}

    for partition in LP.values():
        LP_sizes[partition] = LP_sizes.get(partition, 0) + 1
    
    return {"LP_map": LP, "LP_sizes": LP_sizes}


def user_graph_stats(user_id, toxic_users, SGs, centralities, CCs, LPs):
    metrics = {}

    for SG_name, SG in SGs.items():

        # centralities
        for cent_name, cent_map in centralities[SG_name]:
            cent_key = f"{SG_name}_u_cent_{cent_name}"
            metrics[cent_key] = cent_map[user_id]

        # directed 
        if SG.is_directed():
            # edges in & out | toxic
            edges_in = set(SG.predecessors(user_id))
            edges_out = set(SG.successors(user_id))
            edges_2way = edges_in & edges_out

            edges_in_n_tox = len(edges_in & toxic_users)
            edges_out_n_tox = len(edges_out & toxic_users)
            edges_2way_n_tox = len(edges_2way & toxic_users)

            edges_in_f_tox = safe_div(edges_in_n_tox, len(edges_in))
            edges_out_f_tox = safe_div(edges_out_n_tox, len(edges_out))
            edges_2way_f_tox = safe_div(edges_2way_n_tox, len(edges_2way))

            metrics.update({
                f"{SG_name}_edges_in_n": len(edges_in),
                f"{SG_name}_edges_in_n_tox": edges_in_n_tox,
                f"{SG_name}_edges_in_f_tox": edges_in_f_tox,
                f"{SG_name}_edges_out_n": len(edges_out),
                f"{SG_name}_edges_out_n_tox": edges_out_n_tox,
                f"{SG_name}_edges_out_f_tox": edges_out_f_tox,
                f"{SG_name}_edges_2way_n": len(edges_2way),
                f"{SG_name}_edges_2way_n_tox": edges_2way_n_tox,
                f"{SG_name}_edges_2way_f_tox": edges_2way_f_tox                    
            })

        # undirected
        else:
            # edges | toxic
            edges = set(SG.neighbors(user_id))
            edges_n_tox = len(edges & toxic_users)
            edges_f_tox = safe_div(edges_n_tox, len(edges))

            # CCs
            user_cc = CCs[SG_name]["CC_map"][user_id]
            cc_size_n = CCs[SG_name]["CC_sizes"][user_cc]
            cc_size_f = cc_size_n / SG.number_of_nodes()

            # Louvain paritions
            user_lp = LPs[SG_name]["LP_map"][user_id]
            lp_size_n = LPs[SG_name]["LP_sizes"][user_lp]
            lp_size_f = lp_size_n / SG.number_of_nodes()

            metrics.update({
                f"{SG_name}_edges_n": len(edges),
                f"{SG_name}_edges_n_tox": edges_n_tox,
                f"{SG_name}_edges_f_tox": edges_f_tox,
                f"{SG_name}_user_cc_size_n": cc_size_n,
                f"{SG_name}_user_cc_size_f": cc_size_f,
                f"{SG_name}_user_lp_size_n": lp_size_n,
                f"{SG_name}_user_lp_size_f": lp_size_f
            })
            
    return metrics


def embeddedness_stats(
        user_id, root_user_id, user_ids, toxic_users,
        net_embs, user_id_to_idx, user_n_friends, 
        SG_follow_di, SG_reply_di
    ):
    # NB: excludes the root from all metrics
    
    user_idx = user_id_to_idx[user_id]

    etypes = [
        "all", "toxic", "non-toxic",
        "follow_no", "follow_in", "follow_out", "follow_in_out", "follow_both",
        "reply_no", "reply_in", "reply_out", "reply_in_out", "reply_both"
    ]

    # fetch in/out edges of follow and reply graphs    
    follow_edges_in = set(SG_follow_di.predecessors(user_id))
    follow_edges_out = set(SG_follow_di.successors(user_id))
    
    reply_edges_in = set(SG_reply_di.predecessors(user_id))
    reply_edges_out = set(SG_reply_di.successors(user_id))
    
    # accum vars
    g_n_common = {i: [] for i in etypes}
    g_f_common_union = {i: [] for i in etypes}
    g_f_common_min = {i: [] for i in etypes}

    u_n_friends = user_n_friends[user_idx]

    for alter_user_id in user_ids:
        if (alter_user_id == root_user_id or 
            alter_user_id == user_id or
            alter_user_id not in user_id_to_idx
           ):
            continue

        alter_user_idx = user_id_to_idx[alter_user_id]
        alter_n_friends = user_n_friends[alter_user_idx]

        user_idx_pair = tuple(sorted([user_idx, alter_user_idx]))
        n_common = net_embs.get(user_idx_pair, 0)

        n_union = u_n_friends + alter_n_friends - n_common
        f_common_union = n_common / n_union
        f_common_min = n_common / min(u_n_friends, alter_n_friends)

        # follow edge type
        alter_follow_in = alter_user_id in follow_edges_in
        alter_follow_out = alter_user_id in follow_edges_out

        follow_etype = "follow_no"
        if alter_follow_in and alter_follow_out:
            follow_etype = "follow_both"
        elif alter_follow_in:
            follow_etype = "follow_in"
        elif alter_follow_out:
            follow_etype = "follow_out"

        # reply edge type
        alter_reply_in = alter_user_id in reply_edges_in
        alter_reply_out = alter_user_id in reply_edges_out

        reply_etype = "reply_no"
        if alter_reply_in and alter_reply_out:
            reply_etype = "reply_both"
        elif alter_reply_in:
            reply_etype = "reply_in"
        elif alter_reply_out:
            reply_etype = "reply_out"

        # toxicity edge type
        tox_etype = "toxic" if alter_user_id in toxic_users else "non-toxic"

        # all
        g_n_common["all"].append(n_common)
        g_f_common_union["all"].append(f_common_union)
        g_f_common_min["all"].append(f_common_min)

        # update follow edges stats
        g_n_common[follow_etype].append(n_common)
        g_f_common_union[follow_etype].append(f_common_union)
        g_f_common_min[follow_etype].append(f_common_min)

        if follow_etype in ("follow_in", "follow_out"):
            g_n_common["follow_in_out"].append(n_common)
            g_f_common_union["follow_in_out"].append(f_common_union)
            g_f_common_min["follow_in_out"].append(f_common_min)

        # update reply edges stats
        g_n_common[reply_etype].append(n_common)
        g_f_common_union[reply_etype].append(f_common_union)
        g_f_common_min[reply_etype].append(f_common_min)

        if reply_etype in ("reply_in", "reply_out"):
            g_n_common["reply_in_out"].append(n_common)
            g_f_common_union["reply_in_out"].append(f_common_union)
            g_f_common_min["reply_in_out"].append(f_common_min)

        # toxicity
        g_n_common[tox_etype].append(n_common)
        g_f_common_union[tox_etype].append(f_common_union)
        g_f_common_min[tox_etype].append(f_common_min)


    # sanity checks
    assert (len(g_n_common["follow_in"]) + len(g_n_common["follow_both"])) \
            == len(follow_edges_in)
    assert (len(g_n_common["follow_out"]) + len(g_n_common["follow_both"])) \
            == len(follow_edges_out)


    # compute summary stats
    metrics = {}

    for etype in etypes:
        # n_common
        n_common_arr = np.array(g_n_common[etype])
        n_common_1p_arr = n_common_arr[n_common_arr > 0]
        n_common_f1p = safe_div(len(n_common_1p_arr), len(n_common_arr))

        metrics[f"n_common_{etype}_n"] = len(n_common_arr)
        metrics[f"n_common_{etype}_mean"] = safe(np.mean, n_common_arr)
        metrics[f"n_common_{etype}_var"] = safe(np.var, n_common_arr)
        metrics[f"n_common_{etype}_f1p"] = n_common_f1p
        metrics[f"n_common_{etype}_1p_ent"] = safe(entropy, n_common_1p_arr)
        metrics[f"n_common_{etype}_1p_gini"] = safe(gini, n_common_1p_arr)

        # f_common_union
        f_common_union_arr = np.array(g_f_common_union[etype])
        metrics[f"f_common_union_{etype}_mean"] = safe(np.mean, f_common_union_arr)
        metrics[f"f_common_union_{etype}_var"] = safe(np.var, f_common_union_arr)

        # f_common_min
        f_common_min_arr = np.array(g_f_common_min[etype])
        metrics[f"f_common_min_{etype}_mean"] = safe(np.mean, f_common_min_arr)
        metrics[f"f_common_min_{etype}_var"] = safe(np.var, f_common_min_arr)
        
    return metrics


def dyad_stats(
        user_id, p_user_id, p_tweet_id, 
        user_id_to_idx, SGs, centralities, CCs, LPs,
        follow_edges_idxs, user_n_friends, user_n_followers, net_embs,
        reply_dyads_count, reply_dyads_tox_count, user_alignments, 
        toxicity_scores, toxicity_threshold
    ):
    # NB: the parent follow graph might be missing => returning Nones
    
    metrics = {}
    
    # init
    user_idx = user_id_to_idx[user_id]
    p_user_idx = user_id_to_idx.get(p_user_id, None)
    
    # parent tox
    p_tweet_tox = is_toxic(toxicity_scores, p_tweet_id, toxicity_threshold)

    # follow edge type
    if p_user_idx is None:
        follow_edge_type = None
        
    elif ((user_idx, p_user_idx) in follow_edges_idxs and 
          (p_user_idx, user_idx) in follow_edges_idxs):
        follow_edge_type = "O==O"

    elif (p_user_idx, user_idx) in follow_edges_idxs:
        follow_edge_type = "O->O"

    elif (user_idx, p_user_idx) in follow_edges_idxs:
        follow_edge_type = "O<-O"

    else:
        follow_edge_type = "O  O"
        
    # parent <> child: replies & toxic replies
    replies_n_child_parent = reply_dyads_count[(user_id, p_user_id)]
    replies_n_tox_child_parent = reply_dyads_tox_count[(user_id, p_user_id)]
    replies_f_tox_child_parent = safe_div(
        replies_n_tox_child_parent, 
        replies_n_child_parent
    )

    replies_n_parent_child = reply_dyads_count[(p_user_id, user_id)]
    replies_n_tox_parent_child = reply_dyads_tox_count[(p_user_id, user_id)]
    replies_f_tox_parent_child = safe_div(
        replies_n_tox_parent_child, 
        replies_n_parent_child
    )

    replies_f_tox_both = safe_div(
        (replies_n_tox_child_parent + replies_n_tox_parent_child), 
        (replies_n_child_parent + replies_n_parent_child)
    )

    # embeddedness
    n_common, f_common_union, f_common_min = None, None, None    
    
    if p_user_idx is not None:
        user_idx_pair = tuple(sorted([user_idx, p_user_idx]))
        n_common = net_embs.get(user_idx_pair, 0)
        
        n_friends = user_n_friends[user_idx]
        p_n_friends = user_n_friends[p_user_idx]
        
        f_common_union = n_common / (n_friends + p_n_friends - n_common)
        f_common_min = n_common / min(n_friends, p_n_friends)
        
    # delta n_friends/n_followers
    n_friends_d, n_followers_d = None, None
    
    if p_user_idx is not None:
        n_friends_d = user_n_friends[p_user_idx] - user_n_friends[user_idx]

    if (p_user_idx is not None and
        user_n_followers[p_user_idx] is not None and
        user_n_followers[user_idx] is not None):
        # NB: n_followers could be missing even if other net metrics are present
        n_followers_d = user_n_followers[p_user_idx] - user_n_followers[user_idx]
    
    # alignments
    alg_num = user_alignments.get(user_id, None)
    p_alg_num = user_alignments.get(p_user_id, None)
    alg_num_d, alg_cat_same = None, None

    if alg_num is not None and p_alg_num is not None:
        alg_num_d = p_alg_num - alg_num
        alg_cat_same = (alignment_cat(p_alg_num) == alignment_cat(alg_num))

    # subgraph metrics
    for SG_name, SG in SGs.items():
        # nodes might be missing (e.g., follow graph n/a)
        if user_id not in SG or p_user_id not in SG:
            continue
        
        # centralities
        for cent_name, cent_map in centralities[SG_name]:
            cent_d_key = f"{SG_name}_u_cent_d_{cent_name}"
            cent_d_val = cent_map[p_user_id] - cent_map[user_id]
            metrics[cent_d_key] = cent_d_val
        
        # connected components & louvain paritions
        if not SG.is_directed():
            # CCs
            CC_map = CCs[SG_name]["CC_map"]
            CC_sizes = CCs[SG_name]["CC_sizes"]

            user_cc = CC_map[user_id]
            p_user_cc = CC_map[p_user_id]

            user_cc_size = CC_sizes[user_cc]
            p_user_cc_size = CC_sizes[p_user_cc]

            metrics.update({
                f"{SG_name}_same_cc": user_cc == p_user_cc,
                f"{SG_name}_cc_size_d": p_user_cc_size - user_cc_size
            })

            # Louvain paritions
            LP_map = LPs[SG_name]["LP_map"]
            LP_sizes = LPs[SG_name]["LP_sizes"]

            user_lp = LP_map[user_id]
            p_user_lp = LP_map[p_user_id]

            user_lp_size = LP_sizes[user_lp]
            p_user_lp_size = LP_sizes[p_user_lp]

            metrics.update({
                f"{SG_name}_same_lp": user_lp == p_user_lp,
                f"{SG_name}_lp_size_d": p_user_lp_size - user_lp_size
            })
    
    # record all other metrics
    metrics.update({
        "parent_tweet_tox": p_tweet_tox,
        "follow_edge_type": follow_edge_type,
        "replies_n_child_parent": replies_n_child_parent,
        "replies_n_tox_child_parent": replies_n_tox_child_parent,
        "replies_f_tox_child_parent": replies_f_tox_child_parent,
        "replies_n_parent_child": replies_n_parent_child,
        "replies_n_tox_parent_child": replies_n_tox_parent_child, 
        "replies_f_tox_parent_child": replies_f_tox_parent_child,
        "replies_f_tox_both": replies_f_tox_both,
        "embs_n_common": n_common,
        "embs_f_common_union": f_common_union,
        "embs_f_common_min": f_common_min,
        "n_friends_d": n_friends_d,
        "n_followers_d": n_followers_d,        
        "alg_num_d": alg_num_d,
        "alg_cat_same": alg_cat_same
        # + subgraph metrics (centralities, CC, Louvain paritions)
    })
    
    return metrics


def alignment_cat(alg_num):
    if alg_num is None:
        return None
    if alg_num < 0:
        return "L"
    return "R"


def alignment_stats(user_id, user_ids, user_alignments):
    user_alg_num = user_alignments.get(user_id, None)
    user_alg_cat = alignment_cat(user_alg_num)

    users_alg_same = 0
    users_alg_d_sum = 0
    users_alg_n = 0

    if user_alg_num is not None:
        for alter_id in user_ids:
            if alter_id == user_id:
                continue
                
            alter_alg_num = user_alignments.get(alter_id, None)
            alter_alg_cat = alignment_cat(alter_alg_num)

            if alter_alg_num is not None:
                users_alg_same += int(user_alg_cat == alter_alg_cat)    
                users_alg_d_sum += abs(user_alg_num - alter_alg_num)
                users_alg_n += 1
    
    users_alg_d_mean = safe_div(users_alg_d_sum, users_alg_n)
    users_alg_f_same = safe_div(users_alg_same, users_alg_n)
    
    metrics = {
        "users_alg_d_mean": users_alg_d_mean,
        "users_alg_f_same": users_alg_f_same
    }

    return metrics


def process_conversation(json_fpath, in_root_tweet_id, selected_tweet_ids, 
        TOXICITY_THRESHOLD):

    conversation = json.load(gzip.open(json_fpath))
    
    # NB: make sure it is the right file
    assert in_root_tweet_id == conversation["reply_tree"]["tweet"]

    output = []
    
    tweet_info = conversation["tweets"]
    net_info = conversation["network_features"]
    toxicity_scores = conversation["toxicity_scores"]
    user_alignments = conversation["alignment_scores"]
    
    # root info
    root_tweet_id = conversation["reply_tree"]["tweet"]
    root_tweet_type = conversation["root_tweet_type"]
    root_user_id = conversation["tweets"][root_tweet_id]["user_id"]
    
    # tweets in chronological order
    tweets = list(conversation["tweets"].values())
    tweets.sort(key=lambda x: x["time"])
    
    # get inReplyTo links (child_tweet_id => parent_tweet_id)
    tweet_replyto_id = get_tree_tweet_edges(conversation["reply_tree"])
    tweet_replyto_id = {c_t_id: p_t_id for p_t_id, c_t_id in tweet_replyto_id}
    tweet_replyto_id[root_tweet_id] = None
    
    # user id => idx
    user_id_to_idx = enumerate(net_info["user_ids"])
    user_id_to_idx = {u_id: u_idx for u_idx, u_id in user_id_to_idx}
    
    # n_friends / n_followers (lists), user idx -> n_friends/n_followers
    user_n_friends = net_info["n_friends"]
    user_n_followers = net_info["n_followers"]
    
    # follow edges (user idx, not id!): raw edges 
    follow_edges_idxs = conversation["network_features"]["network"]
    follow_edges_idxs = set(tuple(ij) for ij in follow_edges_idxs)    
    
    # embeddedness: (user_idx1, user_idx2) => n_common_friends
    net_info["network_intersections"] = {(u, v): emb \
        for u, v, emb in net_info["network_intersections"]}
    net_embs = net_info["network_intersections"]

    # follow networks
    G_follow_di = follow_json_to_nx_graph(
        conversation, 
        directed=True, 
        remove_root=True
    )
    G_follow_ud = G_follow_di.to_undirected(reciprocal=False, as_view=False)
    
    # reply networks
    G_reply_di = tree_to_nx_user_graph(
        conversation, 
        directed=True, 
        remove_root=True
    )
    G_reply_ud = G_reply_di.to_undirected(reciprocal=False, as_view=False)
    
    # init tree
    tree = treelib.Tree()
    
    # set of current user ids
    user_ids = set()  
    
    # counters
    n_replies = 0
    n_tox_replies = 0
    
    # replies sent / user | user id => n replies
    user_from_n_replies = defaultdict(int)
    user_from_n_tox_replies = defaultdict(int)
    
    # replies received / user | user id => n replies
    user_to_n_replies = defaultdict(int)
    user_to_n_tox_replies = defaultdict(int)
    
    # (user_id1 --replied_to--> user_id2) 
    reply_dyads_count = defaultdict(int)
    reply_dyads_tox_count = defaultdict(int)
    
    # toxic users: set of users with at least one toxic tweet
    toxic_users = set()

    # LOOP
    for tweet in tweets:
        # tweet / user
        tweet_id = tweet["id"]
        tweet_tox = is_toxic(toxicity_scores, tweet_id, TOXICITY_THRESHOLD)
        tweet_tox_score = toxicity_scores.get(tweet_id, None)
        user_id = tweet["user_id"]
        user_idx = user_id_to_idx.get(user_id, None)
        
        # parent
        parent_tweet_id = tweet_replyto_id[tweet_id]
        parent_user_id = None
        if parent_tweet_id is not None:
            parent_user_id = tweet_info[parent_tweet_id]["user_id"]
        
        # update subgraph users set
        user_ids.add(user_id)
        
        # update reply tree
        tree.create_node(
            identifier=tweet_id, 
            parent=tweet_replyto_id[tweet_id]
        )
        
        # NB: process ONLY the sample tweets
        SKIP = tweet_id not in selected_tweet_ids

        if not SKIP:
            # sanity checks
            assert user_id != root_user_id      # tweet NOT by the root
            assert user_id != parent_user_id    # NOT a self-reply
            assert user_id in user_id_to_idx    # user has follow net info
            assert tree.depth(tweet_id) >= 2    # NOT a direct reply to the root
            assert tweet_tox_score < 0.25 or tweet_tox_score > 0.75

            # n_friends / n_followers stats
            n_friends = user_n_friends[user_idx]
            n_followers = user_n_followers[user_idx]
            n_friends_followers_ratio = None
            if n_followers is not None:
                n_friends_followers_ratio = safe_div(n_friends, n_followers)

            # subgraphs => centralities, CCs, louvain paritions
            # NB: include the current user
            SGs = {
                "follow_di": G_follow_di.subgraph(user_ids),
                "follow_ud": G_follow_ud.subgraph(user_ids),
                "reply_di": G_reply_di.subgraph(user_ids),
                "reply_ud": G_reply_ud.subgraph(user_ids)
            }

            # {sg_name1: {"cent1": {u_id: cent}, "cent2" ...}, sg_name2 ...
            centralities = {}

            # {sg_name1: {"CC_map": {u_id: CC_n}, "CC_sizes": {CC_n: CC_size}}}
            CCs = {}

            # {sg_name1: {"LP_map": {u_id: LP_n}, "LP_sizes": {LP_n: LP_size}}}
            LPs = {}

            for SG_name, SG in SGs.items():
                centralities[SG_name] = compute_centralities(SG)
                if not SG.is_directed():
                    CCs[SG_name] = compute_connected_components(SG)
                    LPs[SG_name] = compute_louvain_partitions(SG)

            # user graph metrics
            user_graph_metrics = user_graph_stats(
                user_id, 
                toxic_users, 
                SGs,
                centralities, 
                CCs, 
                LPs
            )

            # dyad metrics
            user_parent_metrics = dyad_stats(
                user_id, parent_user_id, parent_tweet_id, 
                user_id_to_idx, SGs, centralities, CCs, LPs,
                follow_edges_idxs, user_n_friends, user_n_followers, net_embs,
                reply_dyads_count, reply_dyads_tox_count, user_alignments, 
                toxicity_scores, TOXICITY_THRESHOLD
            )
            user_parent_metrics = {f"dyad_up_{k}": v \
                                   for k, v in user_parent_metrics.items()}
            
            user_root_metrics = dyad_stats(
                user_id, root_user_id, root_tweet_id, 
                user_id_to_idx, SGs, centralities, CCs, LPs,
                follow_edges_idxs, user_n_friends, user_n_followers, net_embs,
                reply_dyads_count, reply_dyads_tox_count, user_alignments, 
                toxicity_scores, TOXICITY_THRESHOLD
            )
            user_root_metrics = {f"dyad_ur_{k}": v \
                                 for k, v in user_root_metrics.items()}
            
            # embeddedness
            emb_metrics = embeddedness_stats(
                user_id, root_user_id, user_ids, toxic_users,
                net_embs, user_id_to_idx, user_n_friends, 
                SGs["follow_di"], SGs["reply_di"]
            )

            # tree metrics
            tree_metrics = tree_stats(tweet_id, tree)

            # alignments
            alignment_metrics = alignment_stats(
                user_id, 
                user_ids, 
                user_alignments
            )
            
            # replies stats (excluding the current reply)
            f_tox_replies = safe_div(n_tox_replies, n_replies)
            
            u_from_n_replies = user_from_n_replies[user_id]
            u_from_n_tox_replies = user_from_n_tox_replies[user_id]
            u_from_f_tox_replies = safe_div(u_from_n_tox_replies, u_from_n_replies)
            
            u_to_n_replies = user_to_n_replies[user_id]
            u_to_n_tox_replies = user_to_n_tox_replies[user_id]
            u_to_f_tox_replies = safe_div(u_to_n_tox_replies, u_to_n_replies)
            
            
            metrics = {
                "tweet_id": tweet_id,
                "root_tweet_id": root_tweet_id, 
                "root_tweet_type": root_tweet_type,
                "conv_n_replies": n_replies,
                "conv_n_tox_replies": n_tox_replies,
                "conv_f_tox_replies": f_tox_replies,
                "user_from_n_replies": u_from_n_replies,
                "user_from_n_tox_replies": u_from_n_tox_replies,
                "user_from_f_tox_replies": u_from_f_tox_replies,
                "user_to_n_replies": u_to_n_replies,
                "user_to_n_tox_replies": u_to_n_tox_replies,
                "user_to_f_tox_replies": u_to_f_tox_replies,
                "n_friends": n_friends,
                "n_followers": n_followers,
                "n_friends_followers_ratio": n_friends_followers_ratio,
                **user_graph_metrics,
                **user_parent_metrics,
                **user_root_metrics,
                **emb_metrics,
                **tree_metrics,
                **alignment_metrics,
                "tweet_tox_score": tweet_tox_score,
                "tweet_tox": tweet_tox,
            }
            
            output.append(metrics)
            
        # update statistics
        # NB: many of these include toxicity stats
        # => updating after computing the reply stats to avoid data snooping
        tweet_tox_int = int(tweet_tox == True)
        
        n_replies += 1
        n_tox_replies += tweet_tox_int
        
        user_from_n_replies[user_id] += 1
        user_from_n_tox_replies[user_id] += tweet_tox_int

        if tweet_tox == True:
            toxic_users.add(user_id)
        
        if parent_user_id is not None:
            user_to_n_replies[parent_user_id] += 1
            user_to_n_tox_replies[parent_user_id] += tweet_tox_int
            
            reply_dyads_count[(user_id, parent_user_id)] += 1
            reply_dyads_tox_count[(user_id, parent_user_id)] += tweet_tox_int
    
    # sanity check
    assert len(output) == len(selected_tweet_ids)

    return output

#
# MAIN
#
def compute_next_reply_metrics(dataset, n_jobs, limit):

    # NB: not as import since all tweets need to have tox < 0.25 or tox > 0.75
    tox_threshold = 0.531

    print("--- Next Reply Metrics ---")
    print(f"Dataset: {dataset}")
    print(f"Num Jobs: {n_jobs}")
    print(f"Limit: {limit}")
    print("----------------------------")

    conf = Config(dataset)
    
    nr_dir = f"{conf.data_root}/next_reply_metrics"
    samples_info_fpath = f"{nr_dir}/{dataset}_paired_sample_tweet_ids.json.gz"
    output_json_fpath = f"{nr_dir}/{dataset}_paired_sample.json.gz"
    output_csv_fpath = f"{nr_dir}/{dataset}_paired_sample.csv"

    # load (fname, root_tweet_id, tweet_ids) triple & add dir path
    samples_info_raw = json.load(gzip.open(samples_info_fpath))
    
    samples_info = []

    for fname, root_tweet_id, tweet_ids in samples_info_raw:
        fpath = f"{conf.conversations_jsons_dir}/{fname}"
        tweet_ids = set(tweet_ids)
        samples_info.append((fpath, root_tweet_id, tweet_ids))

    # go up to limit
    if limit is not None:
        samples_info = samples_info[:limit]

    # compute metrics
    print("Computing metrics ...")

    if n_jobs == 1:
        metrics = [
            process_conversation(
                json_fpath, 
                root_tweet_id, 
                tweet_ids, 
                tox_threshold
            )
            for json_fpath, root_tweet_id, tweet_ids in tqdm(samples_info)
        ]
    else:
        parallel = Parallel(n_jobs=n_jobs, verbose=10, batch_size=1)
        metrics = parallel(
            delayed(process_conversation)(
                    json_fpath, 
                    root_tweet_id, 
                    tweet_ids,
                    tox_threshold
                ) \
                for json_fpath, root_tweet_id, tweet_ids in samples_info
            )

    # flatten the results
    metrics = list(itertools.chain.from_iterable(metrics))
    print(len(metrics))
    
    # output to JSON
    json.dump(metrics, gzip.open(output_json_fpath, "wt"), indent=2)

    # output to CSV
    # (make sure that all metrics have the same set of keys)
    fields = set()
    for m in metrics:
        fields.update(m.keys())

    for m in metrics:
        for f in fields:
            m[f] = m.get(f, None)

    # write_dicts_to_csv(metrics, output_csv_fpath)

    print("Done!")


@click.command()
@click.option('--dataset', required=True,
    type=click.Choice(["news", "midterms"]))
@click.option('--n_jobs', required=True, type=int)
@click.option('--limit', default=None, type=int)
def main(dataset, n_jobs, limit):
    
    compute_next_reply_metrics(
        dataset, 
        n_jobs, 
        limit
    )


if __name__ == "__main__":
    main()

# END
