
"""
Assortativity index

Ported from the NetworkX source code 
https://github.com/networkx/networkx/tree/0cd14f622daa0b70dcb1b8f3404e70383354fceb/networkx/algorithms/assortativity

Passing only an iterator over pairs of attributes instead of a NetworkX graph.
This allows for computing assortativity coefficients without actually 
constructing a graph structure.

Reference:
M. E. J. Newman, Mixing patterns in networks, Physical Review E, 67 026126, 2003

NOTES:
 - numeric AC is just pearson corr, thus it's better to simply use 
 the pearson corr function since it does not create a huge numpy matrix 
 (if the max number is large) and it allows for non-int attributes.
 
"""

import numpy as np
from scipy.stats import pearsonr
from networkx.utils import dict_to_numpy_array



def numeric_attribute_correlation(G, attr_map_1, attr_map_2):
    """
    Computes the correlation (i.e., assortativity) between connected nodes
    in the graph of attr_1 and attr_2.
    
    NB: In undirected networks counts each edge twice.
        That's what Newman recommends in his paper and what networkx does.
    
    Input:
        - G: networkx graph
        - attr_map_1: dict as node_id => numeric attribute value
        - attr_map_2: dict as node_id => numeric attribute value
    Output:
        - Pearson correlation (None if not enough data points)
    """
    # get all pairs
    attr_1 = []
    attr_2 = []
    
    for i_id, i_adj in G.adj.items():
        for j_id, _ in i_adj.items():
            if attr_map_1.get(i_id, None) is None:
                continue

            if attr_map_2.get(j_id, None) is None:
                continue

            attr_1.append(attr_map_1[i_id])
            attr_2.append(attr_map_2[j_id])
    
    return safe_pearsonr(attr_1, attr_2)


def safe_pearsonr(x, y):    
    # at least 2 points needed
    if min(len(x), len(y)) < 2:
        return None
    
    # each array has to have some variation
    if min(len(np.unique(x)), len(np.unique(y))) < 2:
        return None
    
    r, pval = pearsonr(x, y)
    
    if np.isnan(r):
        return None
    
    return r


def categorical_attribute_correlation(G, attr_map_1, attr_map_2, categories=[]):
    attr_iter = categorical_attribute_iter(G, attr_map_1, attr_map_2)
    corr = categorical_attribute_correlation_iter(attr_iter, categories)
    return corr


def categorical_attribute_iter(G, attr_map_1, attr_map_2):
    for i_id, i_adj in G.adj.items():
        for j_id, _ in i_adj.items():
            if attr_map_1.get(i_id, None) is None:
                continue
            if attr_map_2.get(j_id, None) is None:
                continue
            yield (attr_map_1[i_id], attr_map_2[j_id])
            

def categorical_attribute_correlation_iter(xy_iter, categories=[]):

    M_dict = mixing_dict(xy_iter)
    
    # NB: must count all categories
    for c in categories:
        if c not in M_dict:
            M_dict[c] = {}
            
    # convert it to numpy array
    M = dict_to_numpy_array(M_dict)
    
    # must have at least 2 observations
    if M.sum() < 2:
        return None
    
    corr = attribute_ac(M)
    
    if np.isnan(corr):
        return None    
    
    return corr


def mixing_dict(xy, normalized=False):
    """Returns a dictionary representation of mixing matrix.
    Parameters
    ----------
    xy : list or container of two-tuples
       Pairs of (x,y) items.
    normalized : bool (default=False)
       Return counts if False or probabilities if True.
    Returns
    -------
    d: dictionary
       Counts or Joint probability of occurrence of values in xy.
    """
    d = {}
    psum = 0.0
    for x, y in xy:
        if x not in d:
            d[x] = {}
        if y not in d:
            d[y] = {}
        v = d[x].get(y, 0)
        d[x][y] = v + 1
        psum += 1

    if normalized:
        for k, jdict in d.items():
            for j in jdict:
                jdict[j] /= psum
    return d


def attribute_ac(M):
    """
    This computes Eq. (2), (trace(e)-sum(e^2))/(1-sum(e^2)),
    where e is the joint probability distribution (mixing matrix)
    of the specified attribute.
    """
    if M.sum() != 1.0:
        M = M / float(M.sum())
    M = np.asmatrix(M)
    s = (M * M).sum()
    t = M.trace()
    r = (t - s) / (1 - s)
    return float(r)

