from GA.population import Population

import ipdb

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# GA parameters
n_interactions = 2
num_start_indiv = 50000 #Start with a large number so that we can get a good pareto front
num_individuals_per_gen = 2000 #Then reduce to a more reasonable number
n_generations = 100
base_feature_ratio = 0.01
feature_scaling_drop = 1.0

# Misc parameters (including seed)
init_seed = 99
gen_print_freq = 5
percentile_perf_diff_cutoff = 10
critical_feat_percentile = 10


# Data params
use_fake = False

if(use_fake):
    # Make classification parameters
    # (fake data)
    n_samps = 200
    n_feats = 3000
    n_informative_feats = 50
    n_redundant_feats = 10
    n_repeated_feats = 0

    # Without shuffling, we know that the first n_informative + n_redundant + n_repeated
    # features are theoretically important 
    # 0 : n_informative_feats + n_redundant + n_repeated
    # n_redundant = 2 by default, n_repeated = 0 by default
    X, y = make_classification(n_samps, n_feats, 
                            n_informative=n_informative_feats,
                            n_redundant=n_redundant_feats,
                            n_repeated=n_repeated_feats,
                            random_state=init_seed, shuffle=False)

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # Note, make_clasification makes the features the same scale by default
    # but a real dataset probably won't be like this and we will need to scale
    # We should only use the training dataset for scaling.
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=init_seed)

else:
    # Using real data
    # Need to define X_train, X_test, y_train, y_test
    train_x_file = './data/AD_training_dataset_log2.txt'
    test_x_file = './data/AD_testing_dataset_log2.txt'
    all_y_file = './data/AD_demographics.csv'

    X_train = pd.read_csv(train_x_file, sep='\t', index_col=0).transpose()
    X_test = pd.read_csv(test_x_file, sep='\t', index_col=0).transpose()

    y_all = pd.read_csv(all_y_file, index_col=0)
    y_all['Group'] = y_all['Group'].replace({'AD': 1, 'control': 0})

    y_train = y_all.loc[X_train.index, 'Group'].values
    y_test = y_all.loc[X_test.index, 'Group'].values

    col_ind_name_map = X_train.columns

    X_train = X_train.values
    X_test = X_test.values

    # Scale the data here using sklearn StandardScaler
    # fit on the training dataset and then apply to the testing dataset
    # (needed because we use regularization in the models we use to evaluate)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    n_feats = X_train.shape[1]


# Make and initially seed population (outside loop)
ga_pop = Population(base_seed=9, num_features=n_feats, 
                    interaction_num=n_interactions,
                    problem_type='classification',
                    base_feature_ratio=base_feature_ratio,
                    feature_scaling_drop=feature_scaling_drop)
ga_pop.seed_population(num_individuals=num_start_indiv)

# Evaluate zeroth generation (of many more individuals than each generation)
ga_pop.evaluate_current_individuals(X_train, y_train)

# Create new individuals and then evaluate
# then repeat in a loop to continue producing more individuals
for gen_num in tqdm(range(n_generations), desc='Generations'):
    if(not tqdm_avail):
        print(f"Starting Generation {gen_num+1} / {n_generations} ... ", end='', flush=True)

    ga_pop.create_new_generation(num_individuals_per_gen)
    ga_pop.evaluate_current_individuals(X_train, y_train)

    if gen_num % gen_print_freq == 0:
        best_pareto_hashes = ga_pop.pareto_individual_hashes
        str_rep = repr(ga_pop.get_evaluated_individuals_from_hash(best_pareto_hashes)).replace('\n', '\n\t')
        print(f"Best Individuals (Generation {gen_num}):\t{str_rep}")
        print("\n\n-----------------------\n\n")
    
    if(not tqdm_avail):
        print(f"Completed!", flush=True)


# Get the top 10000 performing individuals on the training dataset
# and the best pareto individuals (represented by hashes)
topk_scorer_hashes = ga_pop.get_topk_scoring_individuals(10000)
best_pareto_hashes = ga_pop.pareto_individual_hashes
unique_indiv_hashlist = list(set(topk_scorer_hashes + best_pareto_hashes))

# Get the original scores of the individuals for later comparison
top_indivs = ga_pop.get_evaluated_individuals_from_hash(unique_indiv_hashlist)
top_indivs_train_scores = np.array([indiv.score for indiv in top_indivs])

# Reevaluate top k and best pareto individuals on the testing dataset
# (Note, this returns a LIST of scores, so we do need to convert it to an np.ndarray)
top_indivs_eval_scores = np.array(ga_pop.reevaluate_individuals_from_hashes(unique_indiv_hashlist, X_test, y_test))

# Filter out individuals that performed very poorly on the testing dataset
# that is, they are likely to have overfit the broader training dataset and 
# are not representative of the features useful to the actual problem
score_diff = top_indivs_train_scores - top_indivs_eval_scores
keep_top_notoverfit = np.flatnonzero(score_diff < np.percentile(score_diff, percentile_perf_diff_cutoff))
keep_top_indivs = [top_indivs[i] for i in keep_top_notoverfit]

chr_feats = [indiv.get_chr_features() for indiv in keep_top_indivs]

for depth_int in range(n_interactions):
    depth_chr_feats = np.concatenate([chr[depth_int] for chr in chr_feats])

    feat_used, feat_counts = np.unique(depth_chr_feats, return_counts=True, axis=0)
    feat_prop = feat_counts / len(keep_top_indivs)

    critical_feat_ind = np.flatnonzero(feat_prop > np.percentile(feat_prop, critical_feat_percentile))
    critical_feat_ind = critical_feat_ind[np.argsort(feat_prop[critical_feat_ind])]
    critical_feats_num = feat_used[critical_feat_ind]

    critical_feats = np.array(col_ind_name_map)[critical_feats_num]

    critical_feat_props = feat_prop[critical_feat_ind]

    for i, feat in enumerate(critical_feats_num):
        print(f"{feat} :: {critical_feats[i]} :: {critical_feat_props[i]}")

    ipdb.set_trace()

ipdb.set_trace()