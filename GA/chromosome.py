import numpy as np

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

    def __len__(self):
        return len(self.features)
    
    def get_features(self, X):
        # Gets the features from X corresponding to the chromosomes
        # then multiplies along the last dimension to model interactions
        # for each sample (in a vectorized way)
        return X[:, self.features].prod(axis=-1)
    
    def __repr__(self):
        arr_feat_str = ',\n\t'.join(str(feat) for feat in self.features)
        return f"Chromosome( array([ \n\t{arr_feat_str} ])\n)"
    