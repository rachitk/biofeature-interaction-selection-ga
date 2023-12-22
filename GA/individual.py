from typing import List

from .chromosome import Chromosome
from .utils import softmax, RNG_MAX_INT

from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import ipdb


# Class that stores information about an individual
# and allows one to evaluate that individual
class Individual:
    def __init__(self, chromosomes: List[Chromosome] = []):
        self.chromosomes = chromosomes
        self.hash: tuple[str] = tuple(chr.hash for chr in self.chromosomes)
        self.stats = None
        self.evaluated = False

    def get_chr_sizes(self):
        return [len(chr) for chr in self.chromosomes]
    
    def get_total_size(self):
        return sum(self.get_chr_sizes())
    
    def get_scaled_coef_weights(self, X):
        # Absolute value of coefficient scaled by feature mean value
        # using softmax function implemented in numpy
        # all features scaled relative to each other with no regard
        # for whether they are in the same chromosome or not
        coef_weights_by_chr = []
        scaled_weights = softmax(np.abs(self.fitted_model.coef_).sum(axis=0) * np.abs(X.mean(axis=0)))

        # Note, we need to split by chromosome lengths to make it
        # easier to index into the scaled_weights array
        prev_len = 0
        for chr_len in self.get_chr_sizes():
            coef_weights_by_chr.append(scaled_weights[prev_len:chr_len+prev_len])
            prev_len += chr_len

        return coef_weights_by_chr
    
    # TODO: see if there are any speedups that can be made here
    # since this is the slowest part of every generation
    # (probably will accept some tradeoff here)
    # (perhaps compute subset of X and y beforehand and pass in)
    @ignore_warnings(category=ConvergenceWarning)
    def evaluate(self, X, y, model, score_func, seed=None, index_map=None):
        if(self.evaluated):
            return
        
        rng = np.random.default_rng(seed)
        sklearn_seed = rng.integers(RNG_MAX_INT)

        subset_X = self.subset_construct_features(X, index_map)
        X_train, X_test, y_train, y_test = train_test_split(subset_X, y,
                                                            random_state=sklearn_seed,
                                                            train_size=0.5)

        model = model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        self.score = score_func(y_test, y_pred)
        self.fitted_model = model
        self.evaluated = True
        self.coef_weights = self.get_scaled_coef_weights(subset_X)

        # Note, this is currently broken if we use MSE as the score function
        # TODO: fix for regression problems which do use MSE (make them use negative MSE)
        # self.stats = np.array([*np.log10(np.array(self.get_chr_sizes())+1), 
        #                            -self.score])
        self.stats = np.array([np.log10(self.get_total_size()), 
                               -self.score])
        
        return self
        

    def subset_construct_features(self, X, index_map=None):
        X_feats = [chr.subset_data(X, index_map) for chr in self.chromosomes]
        return np.concatenate(X_feats, axis=-1)

    def get_stats(self):
        return self.stats

    def __repr__(self):
        chr_string = '\n'.join(repr(chr) for chr in self.chromosomes).replace('\n', '\n\t\t')
        return f"Individual(\n\thash: {self.hash},\n\tstats: {self.stats},\n\tchromosomes:\n\t\t{chr_string}, \n)"
    
    def get_chr_features(self, masks=None):
        if(masks is None):
            return [chr.features for chr in self.chromosomes]
        else:
            return [chr.features[mask] for chr, mask in zip(self.chromosomes, masks)]
        
    def get_unique_chr_features(self):
        return np.unique(np.concatenate([chr.features.flatten() for chr in self.chromosomes]))