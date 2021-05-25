
import networkx



def follow_json_to_nx_graph(conversation_json, directed=False, remove_root=False):
    
    # NB: uses the *user ids*, NOT the user indicies as nodes!
    
    graph_json = conversation_json["network_features"]

    # nodes
    user_ids = graph_json["user_ids"]
    
    # edges
    edges = []

    for u_idx, v_idx in graph_json["network"]:
        u_id = user_ids[u_idx]
        v_id = user_ids[v_idx]
        edges.append((u_id, v_id))

    # filter root
    root_tweet_id = conversation_json["reply_tree"]["tweet"]
    root_user_id = conversation_json["tweets"][root_tweet_id]["user_id"]
    root_user_id_missing = root_user_id in graph_json["missing_user_ids"]
    
    if remove_root and not root_user_id_missing:

        # remove root from nodes
        user_ids = [u_id for u_id in user_ids if u_id != root_user_id]

        # remove root from edges
        edges = [(u, v) for u, v in edges
                if u != root_user_id and v != root_user_id]

    # NB: a few conversations have self-loop (Twitter bug), remove them
    edges = [(u, v) for u, v in edges if u != v]

    # create networkx object
    G = networkx.Graph()

    if directed:
        G = networkx.DiGraph()

    G.add_nodes_from(user_ids)
    G.add_edges_from(edges)

    return G

# END
