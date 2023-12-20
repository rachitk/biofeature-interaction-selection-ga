from typing import List

from .chromosome import Chromosome
from .utils import softmax, RNG_MAX_INT

from sklearn.model_selection import train_test_split
import numpy as np

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
        coef_weights_by_chr = []
        scaled_weights = np.abs(self.fitted_model.coef_).sum(axis=0) * np.abs(X.mean(axis=0))

        # Note, we need to split by chromosome lengths and consider each
        # separately (features differently than feature interactions)
        prev_len = 0
        for chr_len in self.get_chr_sizes():
            coef_weights_by_chr.append(softmax(scaled_weights[prev_len:chr_len+prev_len]))
            prev_len += chr_len

        return coef_weights_by_chr
    
    def evaluate(self, X, y, model, score_func, seed=None):
        if(self.evaluated):
            return
        
        rng = np.random.default_rng(seed)
        sklearn_seed = rng.integers(RNG_MAX_INT)

        subset_X = self.subset_construct_features(X)
        X_train, X_test, y_train, y_test = train_test_split(subset_X, y,
                                                            random_state=sklearn_seed)

        model = model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        self.score = score_func(y_test, y_pred)
        self.fitted_model = model
        self.evaluated = True
        self.coef_weights = self.get_scaled_coef_weights(subset_X)

        self.stats = np.array([*self.get_chr_sizes(), 
                                   -self.score])
        

    def subset_construct_features(self, X):
        X_feats = [chr.subset_data(X) for chr in self.chromosomes]
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