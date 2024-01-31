from typing import Dict, List, Tuple

from .individual import Individual
from .chromosome import Chromosome
from .utils import get_pareto_front, RNG_MAX_INT, JL_VERBOSITY, softmax
from .utils import MODEL_CLASSES, SCORE_FUNCS
from .mutations import add_feature, remove_feature, alter_feature_depth

import numpy as np
from sklearn.base import clone

from joblib import Parallel, delayed

import ipdb


# Make TQDM dependency optional
try:
    from tqdm import tqdm
    # TODO: do we need to check what backend is in use?
    # or will joblib always be able to return a generator?
    def print(*args, **kwargs):
        try:
            return tqdm.write(*args, **kwargs)
        except TypeError:
            args = [str(arg) for arg in args]
            return tqdm.write(*args, **kwargs)
    tqdm_avail = True
except ImportError:
    def tqdm(iterator, *args, **kwargs):
        return iterator
    tqdm_avail = False


# Class that stores an entire population of individuals
# Performs the GA steps (mutating, mating, etc.)
# (TODO: Maybe move the other GA steps to another class)
class Population:
    def __init__(self, base_seed=None, num_features=100, interaction_num=2, eval_num=5,
                 problem_type='classification',
                 base_feature_ratio=0.05,
                 feature_scaling_drop=1.0):
        self.base_seed = base_seed
        self.num_features = num_features
        self.interaction_num = interaction_num
        self.eval_num = eval_num
        self.base_feature_ratio = base_feature_ratio
        self.feature_scaling_drop = feature_scaling_drop

        self.generation_data: List[Dict[str, Tuple[str]]] = []
        self.rng = np.random.default_rng(self.base_seed)

        # Properties with individuals
        self.current_individuals: Dict[Tuple[int], Individual] = {}
        self.evaluated_individuals: Dict[Tuple[int], Individual] = {}
        self.pareto_individual_hashes: List[Tuple[int]] = []

        # Add individual with no features in any of the chromosome
        # to the set of evaluated individuals to avoid having issues
        # when trying to evaluate (since it has no features)
        empty_individual = Individual([Chromosome(np.empty((0, chr_num+1), dtype=int))
                                        for chr_num in range(self.interaction_num)])
        
        # Set its stats to be the worst possible
        # with chromosome lengths of infinity and score of 0.0
        # empty_individual.stats = np.array([np.inf for _ in range(interaction_num+1)])
        # empty_individual.stats[-1] = 0.0
        empty_individual.stats = np.array([np.inf, 0.0])
        empty_individual.score = 0.0
        empty_individual.coef_weights = [np.array([]) for _ in range(interaction_num)]
        empty_individual.evaluated = True
        self.evaluated_individuals[empty_individual.hash] = empty_individual
        self.empty_individual_hash = empty_individual.hash

        # TODO: Allow user to define number of jobs (1 = no parallel)

        # Get problem type and model class/score func that goes with it
        # Note these have an UNSET seed, which needs to be set later
        self.problem_type = problem_type
        self.model_class = MODEL_CLASSES.get(problem_type, None)
        self.score_func = SCORE_FUNCS.get(problem_type, None)

        if(self.model_class is None):
            raise ValueError(f"{self.problem_type} not supported. Only supports `classification` or `regression`.")

        # Check if evaluation number is >1 (if not, throw error since KFold breaks)
        if(self.eval_num < 2):
            raise ValueError(f"Evaluation number must be >=2, but was {self.eval_num}.")


    def seed_population(self, num_individuals=1000,
                        initial_sizes=None, seed=None,
                        add_now=True):
        '''Adds individuals to the population based on requested parameters'''

        # By default use initial sizes of the total number of features
        # and then decrease based on interaction number by a scale factor of self.feature_scaling_drop
        if initial_sizes is None:
            # So if the base ratio is 0.05, then the initial sizes would be:
            # [0.05*num_features, 0.05*num_features/self.feature_scaling_drop, 0.05*num_features/self.feature_scaling_drop^2, ...]
            initial_sizes = [int(self.base_feature_ratio * self.num_features/(self.feature_scaling_drop**i)) for i in range(self.interaction_num)]

        # If a single value is passed, then assume we want to use that value
        # for every single depth of interaction
        elif isinstance(initial_sizes, int):
            initial_sizes = [initial_sizes for _ in range(self.interaction_num)]

        # If a list is passed, then it needs to be the same length as the number
        # of interactions or this won't work
        elif len(initial_sizes) != self.interaction_num:
            raise ValueError(f"Number of initial sizes passed to population was {len(initial_sizes)}, "
                             f"but expected {self.interaction_num} or a single integer!")

        # RNG of this seed is based on the passed seed value only if one is passed
        # otherwise, will use the base population RNG (which may have progressed since instantiation)
        rng = np.random.default_rng(seed) if seed is not None else self.rng

        rng_seeds = rng.integers(RNG_MAX_INT, size=(num_individuals,))
        int_num = self.interaction_num
        n_feat = self.num_features

        # Create individuals with the requested chromosomes
        def create_indiv_job(rng_seed):
            rng_seed = np.random.default_rng(rng_seed)
            return Individual([Chromosome(rng_seed.integers(n_feat, 
                                              size=(initial_sizes[chr_num], chr_num+1))) 
                                              for chr_num in range(int_num)])
        
        # Use shared memory here to avoid serializing the original object
        indiv_list = [r for r in tqdm(Parallel(return_as='generator', n_jobs=-1, verbose=JL_VERBOSITY)
                                      (delayed(create_indiv_job)(rng_seed) for rng_seed in rng_seeds),
                                        total=num_individuals, leave=False, desc="Seeding")]
        
        # indiv_list = [Individual([Chromosome(rng.integers(self.num_features, 
        #                                       size=(initial_sizes[chr_num], chr_num+1))) 
        #                                       for chr_num in range(self.interaction_num)])
        #                         for _ in range(num_individuals)]
        
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


    def evaluate_current_individuals(self, X, y, seed=None):
        '''
        Evaluates individuals within the current set in the population 
        using an ElasticNet for accuracy and based on the number of features
        provided along each chromosome
        '''

        rng = np.random.default_rng(seed) if seed is not None else self.rng
        model_seed = rng.integers(RNG_MAX_INT)

        model_class = clone(self.model_class)
        score_func = self.score_func

        # Apply model seed to this version of the model
        model_class.set_params(random_state=model_seed)

        # Seeds for each of the evaluations (for splits, not for models)
        rng_seeds = rng.integers(RNG_MAX_INT, size=(len(self.current_individuals),))

        # Replace already-evaluated individuals with their evaluated versions
        dupe_hashes = self.current_individuals.keys() & self.evaluated_individuals.keys()
        for dupe_hash in dupe_hashes:
            del self.current_individuals[dupe_hash]

        # Parallelized evaluation scheme
        # TODO: Consider finding a way to precompute the subset of X needed for each Individual
        # so as to avoid having to pass the entire X to each copy of the function
        # through some reshaping and binning magic (get unique features needed across all
        # individuals and then index into that subset of features specifically)
            
        # Unique indices of the features needed across every individual
        # (this will probably be basically all the features in earlier generations
        # but as diversity drops this will hopefully be a [much?] smaller subset)
        unique_features_all = np.unique(np.concatenate([indiv.get_unique_chr_features() for indiv in self.current_individuals.values()]))
        index_map = {feat: i for i, feat in enumerate(unique_features_all)}
        needed_feat_X = X.take(unique_features_all, axis=-1)

        curr_eval_num = self.eval_num
        curr_indiv_num = len(self.current_individuals)
        curr_indivs = self.current_individuals.values()

        def eval_func_job(indiv, rng_seed, eval_num=5):
            return indiv.evaluate(needed_feat_X, y, clone(model_class), score_func, rng_seed, index_map, eval_num)

        eval_indivs = [r for r in tqdm(Parallel(return_as='generator', n_jobs=-1, verbose=JL_VERBOSITY)(delayed(eval_func_job)(indiv, rng_seed, curr_eval_num) 
                                                                        for indiv, rng_seed in 
                                                                        zip(curr_indivs, rng_seeds)), 
                                        total=curr_indiv_num, leave=False, desc="Evaluation")]

        # Assign (parallelization means operations not done in-place on original objects)
        self.current_individuals = {indiv.hash: indiv for indiv in eval_indivs}

        self.evaluated_individuals = {**self.current_individuals, **self.evaluated_individuals}

        # Compute the current pareto front
        self.determine_pareto_best_individuals()

        # Store hashes of the current individuals and pareto best individuals
        # in the generation data
        this_gen_data = {
            'current': list(self.current_individuals.keys()),
            'pareto': self.pareto_individual_hashes
        }

        self.generation_data.append(this_gen_data)

    
    def determine_pareto_best_individuals(self):
        # Get the individuals that are nondominated by Pareto standards
        # as the "pareto best" individuals that make up the Pareto front

        # This can only be run after all individuals in the current generation
        # have been evaluated (or it will not work as they are not in evaluated individuals)
        check_hashes = self.pareto_individual_hashes + list(self.current_individuals.keys())

        stats = np.stack([self.evaluated_individuals[indiv_hash].get_stats() for indiv_hash in check_hashes])
        pareto_best_inds = get_pareto_front(stats)

        # TODO: see if there's a faster/better way to construct
        # this list of tuples from the original list + a numpy array of indices
        self.pareto_individual_hashes = [check_hashes[i] for i in pareto_best_inds]

        return self.pareto_individual_hashes
    

    def get_topk_scoring_individuals(self, k=100):
        # Get the top k scoring individuals (regardless of length)
        # for any downstream purpose (typically evaluating on the testing dataset)

        if k > len(self.evaluated_individuals):
            print(f"Warning: k was set to {k} but only {len(self.evaluated_individuals)} total individuals have been evaluated in this population. "
                  "k will be set to the number of evaluated individuals instead.")
            k = len(self.evaluated_individuals)

        # This can only be run after all individuals in the current generation
        # have been evaluated (or it will not work as they are not in evaluated individuals)
        eval_ind_keys = list(self.evaluated_individuals.keys())
        stats = np.stack([self.evaluated_individuals[hash].get_stats() for hash in eval_ind_keys])

        # Argpartition returns the top k elements in any order
        # Note that we effectively sort by the last column (score) and then get the top k from that
        # (remember that score is negative here, so it will be the bottom k technically)
        top_indices = stats[:,-1].argpartition(-k)[:k]
        topk_indiv_hashes = [eval_ind_keys[i] for i in top_indices]

        return topk_indiv_hashes
    

    def get_atavism_individuals(self, atavism_num=10, seed=None):
        # Get random individuals in the previous evaluated set
        # scaled by their respective accuracies
        # TODO: also scale by lengths? Uniform baseline with accuracies on top of that?
        rng = np.random.default_rng(seed) if seed is not None else self.rng

        # This can only be run after all individuals in the current generation
        # have been evaluated (or it will not work as they are not in evaluated individuals)
        with np.errstate(divide='ignore'):
            # Negative because the GA tries to minimize later on via Pareto, 
            # but we want to scale based on maximizing here
            atavism_probs = softmax(np.log([-indiv.stats[-1] for indiv in self.evaluated_individuals.values()]))
            atavism_indivs = rng.choice(list(self.evaluated_individuals.values()), atavism_num, replace=False, p=atavism_probs)

        return atavism_indivs.tolist()


    def create_new_generation(self, num_individuals=1000):
        # Compute proportions of new individuals, mutated, etc.
        # TODO: make these parameters - either of the population
        # or of the function (hyperparameters)
        new_indiv_num = int(0.10 * num_individuals)
        mate_num = int(0.40 * num_individuals)
        mutate_num = int(0.50 * num_individuals)
        
        # Atavism ratio is defined as ratio of atavistic individuals to pareto individuals
        # atavism here will be adding in randomly selected evaluated individuals for mating/mutation
        # Note that we shouldn't introduce the empty individual here
        # Can set atavism ratio to 0 to disable this
        atavism_ratio = 1.0

        # Create some random individuals to include
        new_rand_indivs = self.seed_population(num_individuals=new_indiv_num, add_now=False, seed=self.rng) 

        # Get the top individuals in the population
        # and then randomly mate/mutate them all
        pareto_indiv_hashes = self.pareto_individual_hashes
        pareto_indivs = [self.evaluated_individuals[pareto_hash] 
                         for pareto_hash in pareto_indiv_hashes]
        
        # Get some people scaled based on accuracy to be parents
        # Accuracy is scaled by the number of people to select, so that
        # accuracy is more valued if there are more people to bring back
        atavism_num = min(int(len(pareto_indivs) * atavism_ratio), 100)
        atavism_indivs = self.get_atavism_individuals(atavism_num=atavism_num, seed=self.rng)

        # Combined pareto individuals and atavism individuals for parents
        parent_indivs = pareto_indivs + atavism_indivs

        # Perform mating of best pareto individuals (and atavism individuals)
        # Will try to make mate_num number of children
        # but may produce some nonunique children who will be merged
        child_indivs = self.mate_individuals(parent_indivs, attempt_num=mate_num, seed=self.rng)

        # Randomly mutate best pareto individuals (and atavism individuals)
        # Will try to make mutate_num number of mutants
        # but may make nonunique mutants that will be merged
        mutant_indivs = self.mutate_individuals(parent_indivs, attempt_num=mutate_num, seed=self.rng)

        # Set the current generation to the new set of individuals
        self.current_individuals = {**new_rand_indivs,
                                    **child_indivs,
                                    **mutant_indivs}

        return self.current_individuals
                

    # Function to provide new individuals through random mating
    def mate_individuals(self, individuals, attempt_num=100, seed=None):
        '''
        individuals: Dict[Tuple[int], Individual]
        returns child_indivs: Dict[Tuple[int], Individual]
        '''

        rng = np.random.default_rng(seed) if seed is not None else self.rng
        
        # Select parents to mate
        pairings = rng.choice(individuals, size=(attempt_num, 2))

        # Sample integers to get RNG seeds 
        # (so that when dispatched, we can ensure determinism)
        rng_seeds = rng.integers(RNG_MAX_INT, size=(attempt_num,))

        # Parallelized mating scheme
        # TODO: Allow user to define number of jobs (1 = no parallel)
        def mate_func_job(p1, p2, pair_rng):
            pair_rng = np.random.default_rng(pair_rng)

            # Get random features from parent 1 half-based on coefficient weights
            # plus a uniform distribution to allow for some random chance
            p1_mask = [pair_rng.uniform(size=csize) < (probs/2 + 0.5) for csize, probs in  
                       zip(p1.get_chr_sizes(), p1.coef_weights)]
            p1_selected = p1.get_chr_features(p1_mask)

            # Get random features from parent 2 half-based on coefficient weights
            # plus a uniform distribution to allow for some random chance
            p2_mask = [pair_rng.uniform(size=csize) < (probs/2 + 0.5) for csize, probs in 
                       zip(p2.get_chr_sizes(), p2.coef_weights)]
            p2_selected = p2.get_chr_features(p2_mask)

            # Merge to get new individual
            return Individual([Chromosome(np.concatenate([p1_chr, p2_chr])) 
                               for p1_chr, p2_chr in zip(p1_selected, p2_selected)])

        child_indivs = [r for r in tqdm(Parallel(return_as='generator', n_jobs=-1, verbose=JL_VERBOSITY)(delayed(mate_func_job)(p1, p2, pair_rng) 
                                                       for (p1, p2), pair_rng in zip(pairings, rng_seeds)),
                                        total=len(pairings), leave=False, desc="Mating")]

        child_indivs = {child.hash: child for child in child_indivs}

        return child_indivs
    

    # Function to provide new individuals through random mutations
    def mutate_individuals(self, individuals, attempt_num=100, seed=None, 
                           mutate_fns=[add_feature, remove_feature, alter_feature_depth], 
                           mutate_probs=[0.4, 0.4, 0.2]):
        '''
        individuals: Dict[Tuple[int], Individual]
        returns mutant_indivs: Dict[Tuple[int], Individual]
        '''

        rng = np.random.default_rng(seed) if seed is not None else self.rng

        # Select individuals to mutate
        originals = rng.choice(individuals, size=(attempt_num,))

        # Decide what type of mutation to apply for each individual
        mutations = rng.choice(mutate_fns, size=(attempt_num,), p=mutate_probs)

        # Sample integers to get RNG seeds 
        # (so that when dispatched, we can ensure determinism)
        rng_seeds = rng.integers(RNG_MAX_INT, size=(attempt_num,))

        pop_metadata = self.get_population_metadata()

        # Can easily dispatch computations to multiple jobs
        def mutate_func_job(indiv, mutation_fn, mute_rng):
            return mutation_fn(indiv, pop_metadata, mute_rng)

        mutant_indivs = [r for r in tqdm(Parallel(return_as='generator', n_jobs=-1, verbose=JL_VERBOSITY)(delayed(mutate_func_job)(indiv, mutation_fn, mute_rng) 
                                                       for indiv, mutation_fn, mute_rng in zip(originals, mutations, rng_seeds)),
                                        total=len(originals), leave=False, desc="Mutation")]

        mutant_indivs = {mutant.hash: mutant for mutant in mutant_indivs}

        return mutant_indivs
    

    # Function that collects population metadata for various functions
    def get_population_metadata(self):
        return {
            'num_features': self.num_features,
            'interaction_num': self.interaction_num,
            'rng': self.rng,
            #'evaluated_hashes': list(self.evaluated_individuals.keys())
        }
    
    
    # Function that takes in a set of hashes and gets those individuals
    def get_evaluated_individuals_from_hash(self, hashes):
        return [self.evaluated_individuals[hash_val] for hash_val in hashes]
    

    # Function that takes in a set of hashes and evaluates those individuals again 
    # on a new dataset, using their best models/coefficients
    def reevaluate_individuals_from_hashes(self, hashes, X, y):
        hash_indivs = self.get_evaluated_individuals_from_hash(hashes)

        # Unique indices of the features needed across every individual
        # (this will probably be basically all the features in earlier generations
        # but as diversity drops this will be a much smaller subset)
        unique_features_all = np.unique(np.concatenate([indiv.get_unique_chr_features() for indiv in hash_indivs]))
        index_map = {feat: i for i, feat in enumerate(unique_features_all)}
        needed_feat_X = X.take(unique_features_all, axis=-1)

        score_func = self.score_func

        def reeval_func_job(indiv):
            pred_y = indiv.apply_model(needed_feat_X, index_map)
            return score_func(y, pred_y)
        
        out_reeval_scores = [r for r in tqdm(Parallel(return_as='generator', n_jobs=-1, verbose=JL_VERBOSITY)(delayed(reeval_func_job)(indiv) 
                                                                        for indiv in hash_indivs), 
                                        total=len(hash_indivs), leave=False, desc="Reevaluation")]
        
        return out_reeval_scores

