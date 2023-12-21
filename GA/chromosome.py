import numpy as np

from .utils import softmax

import ipdb

# Class that stores a chromosome
# Depth of chromosome based on whether was init
# as an interaction or not
class Chromosome:
    def __init__(self, features: np.ndarray):
        self.features = np.unique(np.sort(features, axis=-1), axis=0)
        self.features.setflags(write=False)
        
        # Hash based on byte representation
        self.hash: int = hash(self.features.tobytes())

        # Chromosome "depth" of interaction
        self.depth = self.features.shape[1]

    def __len__(self):
        return self.features.shape[0]
    
    def subset_data(self, X):
        # Gets the features from X corresponding to the chromosomes
        # then multiplies along the last dimension to model interactions
        # for each sample (in a vectorized way)
        return X[:, self.features].prod(axis=-1)
    
    def __repr__(self):
        arr_feat_str = ', '.join(str(feat) for feat in self.features)
        return f"Chromosome( array([ \n\t{arr_feat_str} ])\n)"
    