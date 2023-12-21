import numpy as np
# import numexpr as ne

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
    
    def subset_data(self, X, index_map=None):
        # Gets the features from X corresponding to the chromosomes
        # then multiplies along the last dimension to model interactions
        # for each sample (in a vectorized way)
        if(index_map is not None):
            feat_list = np.vectorize(index_map.__getitem__, otypes=[np.int32])(self.features)
        else:
            feat_list = self.features

        # TODO: see if we can take along each axis and then multiply
        # so that we are essentially multiplying two numpy arrays elementwise
        # Maybe we can use memmapping and store prior products in memory
        # to avoid needing to recompute the product each chromosome 
        
        return X.take(feat_list, axis=-1).prod(axis=-1)
    
        # take_arr = X.take(feat_list, axis=-1)
        # return ne.evaluate("prod(take_arr, axis=1)") # trying to use numexpr
    
    def __repr__(self):
        arr_feat_str = ', '.join(str(feat) for feat in self.features)
        return f"Chromosome( array([ \n\t{arr_feat_str} ])\n)"
    