import numpy as np

from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, mean_squared_error

import ipdb


# Set RNG maximum integer for all other files
RNG_MAX_INT = 2**32 - 1

# Set Joblib verbosity based on whether TQDM is available
try:
    from tqdm import tqdm
    tqdm_avail = True
except ImportError:
    tqdm_avail = False
    
JL_VERBOSITY = not tqdm_avail

# Select model based on problem types
# TODO: decide between ElasticNet and ElasticNetCV for regression
MODEL_CLASSES = {
    'regression': ElasticNetCV(), 
    'classification': LogisticRegressionCV(solver='saga', 
                                            penalty='elasticnet',
                                            l1_ratios=[.1, .5, .9, 1.],
                                            max_iter=100)
}

# Consider f1 score or balanced accuracy for classification
# instead of roc auc score
SCORE_FUNCS = {
    'regression': mean_squared_error, 
    'classification': roc_auc_score
}


# Code below modified from 
# https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
# TODO: allow a threshold/cutoff for considering dominated
# since accuracy is a bit stochastic based on the train/test/val split
# (or maybe synchronize the train/test/val split entirely)
# alternatively, maybe return ranks of some kind
def get_pareto_front(costs, return_mask = False):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient
    

def softmax(x):
    # Softmax implementation in numpy
    # Note that this assumes the input is log-scaled
    if(x.size == 0):
        return np.array([])
    
    max_val = x.max()

    if(np.isneginf(max_val)):
        # If all values are -inf, then we can't use the below
        # so just return a uniform distribution
        # which is what the below would return anyways
        # if the subtraction worked
        return np.ones_like(x) / x.size
    
    numer = np.exp(x - x.max())
    return numer / numer.sum()
