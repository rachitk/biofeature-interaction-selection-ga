from GA.population import Population

import ipdb

import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

try:
    from tqdm import tqdm
    def print(*args, **kwargs):
        try:
            return tqdm.write(*args, **kwargs)
        except TypeError:
            args = [str(arg) for arg in args]
            return tqdm.write(*args, **kwargs)
    tqdm_avail = True
    print("NOTE: TQDM available. Progress bars will be printed.\n")
except ImportError:
    def tqdm(iterator, *args, **kwargs):
        return iterator
    tqdm_avail = False
    print("NOTE: TQDM not available. No progress bars will be printed.\n")

# Make classification parameters
n_samps = 200
n_feats = 3000
n_informative_feats = 10

# GA parameters
n_interactions = 2
num_start_indiv = 50000 #Start with a large number so that we can get a good pareto front
num_individuals_per_gen = 5000 #Then reduce to a more reasonable number
n_generations = 1000
base_feature_ratio = 0.01

# Misc parameters (including seed)
init_seed = 9
gen_print_freq = 5

# Without shuffling, we know that the first n_informative + n_redundant + n_repeated
# features are theoretically important 
# 0 : n_informative_feats + n_redundant + n_repeated
# n_redundant = 2 by default, n_repeated = 0 by default
X, y = make_classification(n_samps, n_feats, n_informative=n_informative_feats,
                           random_state=init_seed, shuffle=False)

X = X.astype(np.float32)
y = y.astype(np.float32)

# Note, make_clasification makes the features the same scale by default
# but a real dataset probably won't be like this and we will need to scale
# We should only use the training dataset for scaling.

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=init_seed)

# Make and initially seed population (outside loop)
initial_sizes = [int(base_feature_ratio * n_feats/(10**i)) for i in range(n_interactions)]

ga_pop = Population(base_seed=9, num_features=n_feats, 
                    interaction_num=n_interactions)
ga_pop.seed_population(num_individuals=num_start_indiv,
                       initial_sizes=initial_sizes)

# Evaluate individuals in the population and create new individuals
# then repeat in a loop to continue producing more individuals
for gen_num in tqdm(range(n_generations), desc='Generations'):
    if(not tqdm_avail):
        print(f"Starting Generation {gen_num+1} / {n_generations} ... ", end='', flush=True)
    
    ga_pop.evaluate_current_individuals(X, y, 'classification')
    ga_pop.create_new_generation(num_individuals_per_gen)
    
    if(not tqdm_avail):
        print(f"Completed!", flush=True)

    if gen_num % gen_print_freq == 0:
        best_pareto_hashes = ga_pop.pareto_individual_hashes
        str_rep = repr(ga_pop.get_evaluated_individuals_from_hash(best_pareto_hashes)).replace('\n', '\n\t')
        print(f"Best Individuals (Generation {gen_num}):\t{str_rep}")
        print("\n\n-----------------------\n\n")

# Evaluate the final generation of individuals
ga_pop.evaluate_current_individuals(X, y, 'classification')
ga_pop.get_pareto_best_individuals()

# Get the best individuals from the final generation
best_pareto_hashes = ga_pop.pareto_individual_hashes
best_individuals = ga_pop.get_evaluated_individuals_from_hash(best_pareto_hashes)

ipdb.set_trace()