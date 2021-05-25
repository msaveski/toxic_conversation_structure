
import numpy as np
from scipy.stats import entropy as scipy_entropy


def safe_div(x, y):
    if y == 0:
        return None
    return x / y


def safe(f, arr):
    if len(arr) > 0:
        return f(arr)
    return None


def h_index(x):
    """
    Adopted from:
        https://scholarmetrics.readthedocs.io/en/latest/_modules/scholarmetrics/metrics.html#hindex
    """
    
    # assumes 1D numpy array
    assert len(x.shape) == 1
    
    n = x.shape[0]
    
    x_rev_sorted = np.sort(x)[::-1]
    
    idx = np.arange(1, n + 1)
    
    h = sum([p <= c for (c, p) in zip(x_rev_sorted, idx)])
    
    return h


def entropy(x):
    # assumes 1D numpy array
    assert len(x.shape) == 1
    
    # all values must be positive
    assert np.min(x) > 0
    
    # make probabilites
    x_p = x / x.sum()
    
    # H = -np.sum(x_p * np.log2(x_p))
    
    H = scipy_entropy(x_p, base=2)
    
    return H


def gini(x):
    """
    Adopted from:
        https://github.com/oliviaguest/gini/blob/master/gini.py
        
    Equation taken from:
        https://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    """
    
    # assumes 1D numpy array
    assert len(x.shape) == 1
    
    # all values must be positive
    assert np.min(x) > 0
    
    n = x.shape[0]
    x_sorted = np.sort(x)
    index = np.arange(1, n + 1)
    
    coef = (np.sum((2 * index - n  - 1) * x_sorted)) / (n * np.sum(x_sorted))
    
    return coef    
