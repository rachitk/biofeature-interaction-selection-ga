from typing import List

from .chromosome import Chromosome
from .utils import softmax, RNG_MAX_INT

from sklearn.model_selection import KFold
from sklearn.base import clone
import numpy as np

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import ipdb


# Class that stores information about an individual
# and allows one to evaluate that individual
class Individual:
    def __init__(self, chromosomes: List[Chromosome] = []):
        self.chromosomes = chromosomes
        self.hash: tuple[int] = tuple(chr.hash for chr in self.chromosomes)
        self.stats = None
        self.evaluated = False

    def get_chr_sizes(self):
        return [len(chr) for chr in self.chromosomes]
    
    def get_total_size(self):
        return sum(self.get_chr_sizes())
    
    def get_scaled_coef_weights(self, X):
        # TODO: change to use SelectFromModel instead
        # of the feature coefficients (especially since
        # coef_ isn't available for all models)

        # Absolute value of coefficient 
        # all features scaled relative to each other with no regard
        # for whether they are in the same chromosome or not

        # Do not scale the coefficients by the feature means (did previously)
        # especially since the features are normalized to N(0,1) and so their means are close to 0
        # and so this will downplay their importances broadly
        coef_weights_by_chr = []
        with np.errstate(divide='ignore'):
            scaled_weights = softmax(np.log(np.abs(self.fitted_model.coef_).sum(axis=0)))

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
    @ignore_warnings(category=ConvergenceWarning)
    def evaluate(self, X, y, model, score_func, seed=None, index_map=None, num_eval=5):
        if(self.evaluated):
            return
        
        rng = np.random.default_rng(seed)
        subset_X = self.subset_construct_features(X, index_map)

        # Fitting/evaluating the model N times and taking the average performance
        # rather than just doing so once, to minimize noise-related issues
        # (though one can just set num_eval=1 to disable this)
        model_scores = [None] * num_eval
        best_model = None # use the coefficient weights from the best model
        best_score = -np.inf

        # Use KFold to get the number of split evaluations
        sklearn_seed = rng.integers(RNG_MAX_INT)
        kf = KFold(n_splits=num_eval, shuffle=True, random_state=sklearn_seed)

        for i, (train_inds, test_inds) in enumerate(kf.split(subset_X, y)):
            cmodel = clone(model)

            X_train, X_test = subset_X[train_inds], subset_X[test_inds]
            y_train, y_test = y[train_inds], y[test_inds]

            cmodel = cmodel.fit(X_train, y_train)
            # TODO: support regression problems by using predict
            # since predict_proba is only present for classification problems
            # (this also only works for binary classification, so consider changing)
            # score-func signature to be model, X_test, y_pred as replacement
            y_pred = cmodel.predict_proba(X_test)[:,1]
            model_scores[i] = score_func(y_test, y_pred)

            if(model_scores[i] > best_score):
                best_model = cmodel
                best_score = model_scores[i]

        self.score = np.mean(model_scores)
        self.fitted_model = best_model
        self.evaluated = True

        # Use the coefficient weights from the BEST model
        # since while this isn't necessarily a good representation of
        # the feature importance, it does give a good representation of
        # which ones might be most useful for the GA to use
        self.coef_weights = self.get_scaled_coef_weights(subset_X)

        # Note, this is currently broken if we use MSE as the score function
        # TODO: fix for regression problems which do use MSE (make them use negative MSE)
        # self.stats = np.array([*np.log10(np.array(self.get_chr_sizes())+1), 
        #                            -self.score])
        self.stats = np.array([np.log10(self.get_total_size()), 
                               -self.score])
        
        return self
    
    def apply_model(self, X, index_map=None):
        # Apply (trained/evaluated) model to a new dataset
        # (e.g. the test set) and return predictions

        if(not self.evaluated):
            raise ValueError("Trying to reevaluate an individual that hasn't been evaluated/trained yet!")
        
        return self.fitted_model.predict(self.subset_construct_features(X, index_map))

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