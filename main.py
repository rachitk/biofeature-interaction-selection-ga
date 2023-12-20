from GA.population import Population

import ipdb

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

n_samps = 2000
n_feats = 1000
n_interactions = 2
num_individuals = 100

n_generations = 10
init_seed = 9

X, y = make_classification(n_samps, n_feats, n_informative=10,
                           random_state=init_seed)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=init_seed)

# Make and initially seed population (outside loop)
ga_pop = Population(base_seed=9, num_features=n_feats, 
                    interaction_num=n_interactions)
ga_pop.seed_population(seed=9, num_individuals=num_individuals,
                       initial_sizes=[10,3])

# Evaluate individuals in the population and create new individuals
# then repeat in a loop to continue producing more individuals
ga_pop.evaluate_current_individuals(X, y, 'classification')
ga_pop.create_new_generation(num_individuals)
ga_pop.evaluate_current_individuals(X, y, 'classification')

best_pareto_hashes = ga_pop.get_pareto_best_individuals()
best_individuals = ga_pop.get_evaluated_individuals_from_hash(best_pareto_hashes)

ipdb.set_trace()