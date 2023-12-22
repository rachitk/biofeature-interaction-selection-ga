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

n_samps = 20000
n_feats = 3000
n_informative_feats = 10
n_interactions = 2
num_start_indiv = 1000
num_individuals_per_gen = 1000

n_generations = 1000
init_seed = 9
gen_print_freq = 5


X, y = make_classification(n_samps, n_feats, n_informative=n_informative_feats,
                           random_state=init_seed)

X = X.astype(np.float32)
y = y.astype(np.float32)

# Note, make_clasification makes the features the same scale by default
# but a real dataset probably won't be like this and we will need to scale
# We should only use the training dataset for scaling.

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=init_seed)

# Make and initially seed population (outside loop)
ga_pop = Population(base_seed=9, num_features=n_feats, 
                    interaction_num=n_interactions)
ga_pop.seed_population(num_individuals=num_start_indiv,
                       initial_sizes=[int(n_feats/10),int(n_feats/100)])

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