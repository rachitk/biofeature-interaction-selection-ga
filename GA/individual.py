from typing import List

from .chromosome import Chromosome

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
        return self.fitted_model.coef_ * X.mean(axis=0)
    
    def evaluate(self, X, y, model, score_func):
        if(self.evaluated):
            return

        subset_X = self.subset_construct_features(X)
        X_train, X_test, y_train, y_test = train_test_split(subset_X, y)

        model = model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        self.score = score_func(y_test, y_pred)
        self.fitted_model = model
        self.evaluated = True
        self.coef_weights = self.get_scaled_coef_weights(subset_X)

        self.stats = np.array([*self.get_chr_sizes(), 
                                   -self.score])
        

    def subset_construct_features(self, X):
        X_feats = [chr.get_features(X) for chr in self.chromosomes]
        return np.concatenate(X_feats, axis=-1)

    def get_stats(self):
        return self.stats

    def __repr__(self):
        chr_string = '\n'.join(repr(chr) for chr in self.chromosomes).replace('\n', '\n\t\t')
        return f"Individual(\n\thash: {self.hash},\n\tstats: {self.stats},\n\tchromosomes:\n\t\t{chr_string}, \n)"