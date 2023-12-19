from GA.population import Population

import ipdb

from sklearn.datasets import make_classification

n_samps = 500
n_feats = 1000
n_interactions = 2
num_individuals = 100

X, y = make_classification(n_samps, n_feats, n_informative=10)

ga_pop = Population(base_seed=9, num_features=n_feats, 
                    interaction_num=n_interactions)

ga_pop.seed_population(seed=9, num_individuals=100,
                       initial_sizes=[10,3])
ga_pop.evaluate_current_individuals(X, y, 'classification')
ga_pop.create_new_generation()

ipdb.set_trace()