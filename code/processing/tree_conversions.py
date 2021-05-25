
import networkx
import treelib



def get_tree_user_edges(tree, tweets):
    """
    Input:
        - tree:
            recursive tree structure
            {tweet: "tweet_id", replies: [ .... ]}
    Output:
        - list of replier, poster user id pairs
          (poster <- replier)
    """

    parent_tweet_id = tree["tweet"]
    parent_user_id = tweets[parent_tweet_id]["user_id"]

    edges_from_children = []
    downstream_edges = []

    for reply in tree["replies"]:
        reply_tweet_id = reply["tweet"]
        reply_user_id = tweets[reply_tweet_id]["user_id"]

        # add an edge from the parent to the reply
        edges_from_children.append((reply_user_id, parent_user_id))

        # recursively get the edges of the child
        downstream_edges += get_tree_user_edges(reply, tweets)

    return edges_from_children + downstream_edges


def tree_to_nx_user_graph(conversation, directed=True, remove_root=False):
    """
    Input:
        - conversation dictionary:
        {
            "tweets": metadata for each tweet in the tree
            "reply_tree": recursive tree structure
        }
    Output:
        - networkx directed graph
    """

    # extract key fields
    tree = conversation["reply_tree"]
    tweets = conversation["tweets"]
    
    # fetch the root user id
    root_tweet_id = tree["tweet"]
    root_user_id = tweets[root_tweet_id]["user_id"]
    
    # nodes: unique user ids
    nodes = {tweet["user_id"] for tweet in tweets.values()}
    
    # edges: list of (replier, poster) pairs
    edges = get_tree_user_edges(tree, tweets)

    # remove self looops
    edges = [(u, v) for u, v in edges if u != v]
        
    # remove the root user, if necessary
    if remove_root:
        nodes = {u_id for u_id in nodes if u_id != root_user_id}
        edges = [(u, v) for u, v in edges 
                 if u != root_user_id and v != root_user_id]
    
    # create networkx graph
    G = networkx.Graph()
    
    if directed:
        G = networkx.DiGraph()
        
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    return G


def get_tree_tweet_edges(tree):
    """
    Input:
        - tree:
            recursive tree structure
            {tweet: "tweet_id", replies: [ .... ]}
    Output:
        - list of parent-child tweet_ids
    """
    parent_tweet_id = tree["tweet"]
    edges_to_children = []
    childrens_edges = []

    for reply in tree["replies"]:
        reply_tweet_id = reply["tweet"]

        # add an edge from the parent to the reply
        edges_to_children.append((parent_tweet_id, reply_tweet_id))

        # recursively get the edges of child
        childrens_edges += get_tree_tweet_edges(reply)

    return edges_to_children + childrens_edges


def tree_json_to_tree_lib(tree_json):

    def add_nodes(subtree, parent_tweet_id):
        # create the (subtree) root node
        tweet_id = subtree["tweet"]
        tweet_data = {
            "user_id": tweets_metadata[tweet_id]["user_id"],
            "time": tweets_metadata[tweet_id]["time"]
        }

        tree.create_node(
            tag=tweet_id,
            identifier=tweet_id,
            parent=parent_tweet_id,
            data=tweet_data
        )

        # loop through the children
        for reply_subtree in subtree["replies"]:
            reply_tweet = reply_subtree["tweet"]
            add_nodes(reply_subtree, parent_tweet_id=tweet_id)

    tree = treelib.Tree()
    tree_dict = tree_json["reply_tree"]
    tweets_metadata = tree_json["tweets"]
    add_nodes(tree_dict, parent_tweet_id=None)

    return tree
