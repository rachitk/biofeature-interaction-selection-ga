from typing import Dict, List, Tuple

from .individual import Individual
from .chromosome import Chromosome
from .utils import get_pareto_front

import numpy as np
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, mean_squared_error
import ipdb


# Class that stores an entire population of individuals
# Performs the GA steps (mutating, mating, etc.)
# (TODO: Maybe move the other GA steps to another class)
class Population:
    def __init__(self, base_seed=None, num_features=100, interaction_num=2):
        self.current_individuals: Dict[Tuple[int], Individual] = {}
        self.evaluated_individuals: Dict[Tuple[int], Individual] = {}
        self.pareto_individual_hashes: List[Tuple[int]] = []
        self.base_seed = base_seed
        self.num_features = num_features
        self.interaction_num = interaction_num
        self.rng = np.random.default_rng(self.base_seed)

    def seed_population(self, num_individuals=1000,
                        initial_sizes=10, seed=None,
                        add_now=True):
        '''Adds individuals to the population based on requested parameters'''

        if(len(self.current_individuals) != 0):
            raise ValueError("Cannot seed a population that already has individuals!")
            # TODO: maybe allow this to be possible by adding new individuals to a current set

        # RNG of this seed is based on the passed seed value only if one is passed
        # otherwise, will use the base population RNG (which may have progressed since instantiation)
        rng = np.random.default_rng(seed) if seed is not None else self.rng

        if isinstance(initial_sizes, int):
            initial_sizes = [initial_sizes for _ in range(self.interaction_num)]

        # TODO: Parallelize the creation of individuals here
        # Create individuals with the requested chromosomes
        indiv_list = [Individual([Chromosome(rng.integers(self.num_features, 
                                              size=(initial_sizes[chr_num], chr_num+1))) 
                                              for chr_num in range(self.interaction_num)])
                                for _ in range(num_individuals)]
        
        indiv_dict = {indiv.hash: indiv for indiv in indiv_list}

        # Add new individuals now if requested, otherwise return them
        if(add_now):
            self._add_to_population(indiv_dict)
        else:
            return indiv_dict


    def _add_to_population(self, new_individuals):
        # Add new individuals to current set, but maintaining current individuals who
        # are already in the new individuals made
        self.current_individuals = {**new_individuals, **self.current_individuals}


    def evaluate_current_individuals(self, X, y, problem_type='regression'):
        '''
        Evaluates individuals within the current set in the population 
        using an ElasticNet for accuracy and based on the number of features
        provided along each chromosome
        '''

        # TODO: decide between ElasticNet and ElasticNetCV for regression
        model_classes = {
            'regression': ElasticNetCV(), 
            'classification': LogisticRegressionCV(solver='saga', 
                                                   penalty='elasticnet',
                                                   l1_ratios=[.1, .5, .9, .99, 1])
        }

        score_funcs = {
            'regression': mean_squared_error, 
            'classification': roc_auc_score
        }

        model_class = model_classes.get(problem_type, None)
        score_func = score_funcs.get(problem_type, None)

        if(model_class is None):
            raise ValueError(f"{problem_type} not supported. Only supports `classification` or `regression`.")

        # TODO: Definitely set up parallel execution here
        # as we can evaluate all individuals simultaneously
        for hash_i, indiv in self.current_individuals.items():
            if(hash_i in self.evaluated_individuals):
                self.current_individuals[hash_i] = self.evaluated_individuals[hash_i]
                continue
            
            indiv.evaluate(X, y, clone(model_class), score_func)

        self.evaluated_individuals = {**self.current_individuals, **self.evaluated_individuals}

    
    def get_pareto_best_individuals(self):
        # This can only be run after all individuals in the current generation
        # have been evaluated (or it will not work as they are not in evaluated individuals)
        check_hashes = self.pareto_individual_hashes + list(self.current_individuals.keys())

        stats = np.stack([self.evaluated_individuals[indiv_hash].get_stats() for indiv_hash in check_hashes])
        pareto_best_inds = get_pareto_front(stats)

        # TODO: see if there's a faster/better way to construct
        # this list of tuples from the original list + a numpy array of indices
        self.pareto_individual_hashes = [check_hashes[i] for i in pareto_best_inds]

        return self.pareto_individual_hashes


    def create_new_generation(self, num_individuals=1000):
        # Compute proportions of new individuals, mutated, etc.
        # TODO: make these parameters - either of the population
        # or of the function (hyperparameters)
        new_indiv_num = int(0.2 * num_individuals)
        mate_num = int(0.3 * num_individuals)
        mutate_num = int(0.5 * num_individuals)

        # Get the top individuals in the population
        # and then randomly mate/mutate them all
        pareto_indiv_hashes = self.get_pareto_best_individuals()
        pareto_indivs = [self.evaluated_individuals[pareto_hash] 
                         for pareto_hash in pareto_indiv_hashes]

        ipdb.set_trace()

        # Perform mating of best pareto individuals
        

        # Clear the population after we've found the pareto dominant ones
        self.current_individuals = {}

        # Create some random individuals to include
        new_rand_indivs = self.seed_population(num_individuals=new_indiv_num, add_now=False)


