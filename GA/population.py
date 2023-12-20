from typing import Dict, List, Tuple

from .individual import Individual
from .chromosome import Chromosome
from .utils import get_pareto_front, RNG_MAX_INT
from .mutations import add_feature, remove_feature, alter_feature_depth

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

        # Add individual with no features in any of the chromosome
        # to the set of evaluated individuals to avoid having issues
        # when trying to evaluate (since it has no features)
        empty_individual = Individual([Chromosome(np.empty((0, chr_num+1), dtype=int))
                                        for chr_num in range(self.interaction_num)])
        
        # Set its stats to be the worst possible
        # with chromosome lengths of infinity and score of infinity
        # empty_individual.stats = np.array([np.inf for _ in range(interaction_num+1)])
        empty_individual.stats = np.array([np.inf, np.inf])
        empty_individual.coef_weights = [np.array([]) for _ in range(interaction_num)]
        empty_individual.evaluated = True
        self.evaluated_individuals[empty_individual.hash] = empty_individual


    def seed_population(self, num_individuals=1000,
                        initial_sizes=10, seed=None,
                        add_now=True):
        '''Adds individuals to the population based on requested parameters'''

        if isinstance(initial_sizes, int):
            initial_sizes = [initial_sizes for _ in range(self.interaction_num)]

        if(len(initial_sizes) != self.interaction_num):
            raise ValueError(f"Number of initial sizes passed to population was {len(initial_sizes)}, "
                             f"but expected {self.interaction_num} or a single integer!")

        # RNG of this seed is based on the passed seed value only if one is passed
        # otherwise, will use the base population RNG (which may have progressed since instantiation)
        rng = np.random.default_rng(seed) if seed is not None else self.rng

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
        # equivalent to, in Python 3.9:
        # self.current_individuals = new_individuals | self.current_individuals


    def evaluate_current_individuals(self, X, y, problem_type='regression', seed=None):
        '''
        Evaluates individuals within the current set in the population 
        using an ElasticNet for accuracy and based on the number of features
        provided along each chromosome
        '''

        rng = np.random.default_rng(seed) if seed is not None else self.rng
        model_seed = rng.integers(RNG_MAX_INT)

        # TODO: decide between ElasticNet and ElasticNetCV for regression
        model_classes = {
            'regression': ElasticNetCV(random_state=model_seed), 
            'classification': LogisticRegressionCV(solver='saga', 
                                                   penalty='elasticnet',
                                                   l1_ratios=[.1, .5, .9, .99, 1],
                                                   random_state=model_seed)
        }

        score_funcs = {
            'regression': mean_squared_error, 
            'classification': roc_auc_score
        }

        model_class = model_classes.get(problem_type, None)
        score_func = score_funcs.get(problem_type, None)

        rng_seeds = rng.integers(RNG_MAX_INT, size=(len(self.current_individuals),))

        if(model_class is None):
            raise ValueError(f"{problem_type} not supported. Only supports `classification` or `regression`.")

        # TODO: Definitely set up parallel execution here
        # as we can evaluate all individuals simultaneously
        for i, (hash_i, indiv) in enumerate(self.current_individuals.items()):
            if(hash_i in self.evaluated_individuals):
                self.current_individuals[hash_i] = self.evaluated_individuals[hash_i]
                continue
            
            indiv.evaluate(X, y, clone(model_class), score_func, rng_seeds[i])

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
        # TODO: atavism_prop = 0.01 
        # atavism here will be adding in random evaluated individuals
        # perhaps also do this if the number of pareto individuals is small
        # relative to the number of new individuals to be made (to reintroduce
        # diversity into the population)
        # Note that we shouldn't introduce the empty individual here

        # Get the top individuals in the population
        # and then randomly mate/mutate them all
        pareto_indiv_hashes = self.get_pareto_best_individuals()
        pareto_indivs = [self.evaluated_individuals[pareto_hash] 
                         for pareto_hash in pareto_indiv_hashes]

        # Perform mating of best pareto individuals 
        # TODO: maybe mix in some random evaluated individuals
        # Will try to make mate_num number of children
        # but may produce some nonunique children who will be merged
        child_indivs = self.mate_individuals(pareto_indivs, attempt_num=mate_num, seed=self.rng)

        # Randomly mutate best pareto individuals
        # TODO: maybe mix in some random evaluated individuals
        # Will try to make mutate_num number of mutants
        # but may again make nonunique mutants that will be merged
        mutant_indivs = self.mutate_individuals(pareto_indivs, attempt_num=mutate_num, seed=self.rng)

        # Create some random individuals to include
        new_rand_indivs = self.seed_population(num_individuals=new_indiv_num, add_now=False, seed=self.rng)

        # Set the current generation to the new set of individuals
        self.current_individuals = {**child_indivs,
                                    **mutant_indivs,
                                    **new_rand_indivs}
                

    # Function to provide new individuals through random mating
    def mate_individuals(self, individuals, attempt_num=100, seed=None):
        '''
        individuals: Dict[Tuple[int], Individual]
        returns child_indivs: Dict[Tuple[int], Individual]
        '''

        child_indivs = {}

        rng = np.random.default_rng(seed) if seed is not None else self.rng
        
        # Select parents to mate
        pairings = rng.choice(individuals, size=(attempt_num, 2))

        # Sample integers to get RNG seeds 
        # (so that when dispatched, we can ensure determinism)
        rng_seeds = rng.integers(RNG_MAX_INT, size=(attempt_num,))


        # TODO: Parallelize the below for creating mated people
        # Can easily dispatch computations to multiple jobs
        # Consider moving the below to a function elsewhere 
        # (and maybe have multiple mating options)
        for (p1, p2), pair_rng in zip(pairings, rng_seeds):
            pair_rng = np.random.default_rng(pair_rng)
            # Get random features from parent 1
            p1_mask = [pair_rng.uniform(size=csize) < probs for csize, probs in 
                       zip(p1.get_chr_sizes(), p1.coef_weights)]
            p1_selected = p1.get_chr_features(p1_mask)

            # Get random features from parent 2
            p2_mask = [pair_rng.uniform(size=csize) < probs for csize, probs in 
                       zip(p2.get_chr_sizes(), p2.coef_weights)]
            p2_selected = p2.get_chr_features(p2_mask)

            # Merge to get new individual
            child = (Individual([Chromosome(np.concatenate([p1_chr, p2_chr])) 
                                 for p1_chr, p2_chr in zip(p1_selected, p2_selected)]))
            
            child_indivs[child.hash] = child

        return child_indivs
    

    # Function to provide new individuals through random mutations
    def mutate_individuals(self, individuals, attempt_num=100, seed=None, 
                           mutate_fns=[add_feature, remove_feature, alter_feature_depth], 
                           mutate_probs=[0.3, 0.3, 0.4]):
        '''
        individuals: Dict[Tuple[int], Individual]
        returns mutant_indivs: Dict[Tuple[int], Individual]
        '''

        mutant_indivs = {}

        rng = np.random.default_rng(seed) if seed is not None else self.rng

        # Select individuals to mutate
        originals = rng.choice(individuals, size=(attempt_num,))

        # Decide what type of mutation to apply for each individual
        mutations = rng.choice(mutate_fns, size=(attempt_num,), p=mutate_probs)

        # Sample integers to get RNG seeds 
        # (so that when dispatched, we can ensure determinism)
        rng_seeds = rng.integers(RNG_MAX_INT, size=(attempt_num,))


        # TODO: Parallelize the below for creating mated people
        # Can easily dispatch computations to multiple jobs
        for indiv, mutation_fn, mute_rng in zip(originals, mutations, rng_seeds):
            # Apply mutation to individual
            mutant = mutation_fn(indiv, self.get_population_metadata(),
                                 mute_rng)
            mutant_indivs[mutant.hash] = mutant

        return mutant_indivs
    

    # Function that collects population metadata for various functions
    def get_population_metadata(self):
        return {
            'num_features': self.num_features,
            'interaction_num': self.interaction_num,
            'rng': self.rng,
            'evaluated_hashes': list(self.evaluated_individuals.keys())
        }
    
    
    # Function that takes in a set of hashes and gets those individuals
    def get_evaluated_individuals_from_hash(self, hashes):
        return [self.evaluated_individuals[hash_val] for hash_val in hashes]