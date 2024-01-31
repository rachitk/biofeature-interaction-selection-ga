from GA.population import Population

import ipdb

import numpy as np
import pandas as pd
import os

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from synthetic_data import make_binary_classification_with_interaction

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
init_seed = 99999
gen_print_freq = 5
percentile_perf_diff_cutoff = 10
critical_feat_percentile = 10


# Data params
use_fake = False
out_dir_prefix = 'fixed_score/out'


# Define out directory
out_dir = f'{out_dir_prefix}_ninter{n_interactions}_nstart{num_start_indiv}_ngen{n_generations}_nindiv{num_individuals_per_gen}'
out_dir += f'_basefeat{base_feature_ratio}_featdrop{feature_scaling_drop}_seed{init_seed}_data{"Synth" if use_fake else "Real"}'
os.makedirs(out_dir, exist_ok=True)

if(use_fake):
    # Make classification parameters
    # (fake data)
    n_samps = 200
    n_feats = 1000
    n_informative_feats = 50
    n_interaction_feats = 10
    n_int_from_info = 5

    # Without shuffling, we know that the first n_informative
    # features are theoretically important 
    # 0 : n_informative_feats 
    # and the interactions are returned separately
    X, y, important_interactions = make_binary_classification_with_interaction(n_samples=n_samps, n_features=n_feats,
                                                       n_informative=n_informative_feats,
                                                       n_interactions=n_interaction_feats,
                                                       n_interactions_from_informative=n_int_from_info,
                                                       random_state=init_seed)

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    # Note, make_classification makes the features the same scale by default
    # but a real dataset probably won't be like this and we will need to scale
    # We should only use the training dataset for scaling.
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=init_seed)
    
    print("Expected interactions are: ")
    print(important_interactions)
    print("\n")

    col_ind_name_map = np.arange(n_feats)    

else:
    # Using real data
    # Need to define X_train, X_test, y_train, y_test
    # train_x_file = './data/AD_training_dataset_ALL_log2.txt'
    # test_x_file = './data/AD_testing_dataset_ALL_log2.txt'
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


# Report
print(f"Results output to: {out_dir}")
print(f"Number of features: {n_feats}")
print(f'Number of train/test samples: {len(y_train)} / {len(y_test)}')

# Make and initially seed population (outside loop)
ga_pop = Population(base_seed=9, num_features=n_feats, 
                    interaction_num=n_interactions,
                    problem_type='classification',
                    base_feature_ratio=base_feature_ratio,
                    feature_scaling_drop=feature_scaling_drop)
ga_pop.seed_population(num_individuals=num_start_indiv)

# Evaluate zeroth generation (of many more individuals than each generation)
ga_pop.evaluate_current_individuals(X_train, y_train)

best_pareto_hashes = ga_pop.pareto_individual_hashes
str_rep = repr(ga_pop.get_evaluated_individuals_from_hash(best_pareto_hashes)).replace('\n', '\n\t')
print(f"Best Individuals (Initial):\t{str_rep}")
print("\n\n-----------------------\n\n")

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


# Get the top 100000 performing individuals on the training dataset
# and the best pareto individuals (represented by hashes)
# (technically could get every evaluated individual, period)
# (but this ensures computational tractability when reevaluating)
    
# TODO: consider only getting individuals from the last 5-10 generations
# instead of the top individuals over all generations

topk_scorer_hashes = list(ga_pop.get_topk_scoring_individuals(100000))
best_pareto_hashes = ga_pop.pareto_individual_hashes
unique_indiv_hashlist = list(set(topk_scorer_hashes + best_pareto_hashes))

# Remove the empty individual (if present)
try:
    empty_hash = ga_pop.empty_individual_hash
    if(empty_hash in unique_indiv_hashlist):
        unique_indiv_hashlist.remove(empty_hash)
except:
    pass

# Get the original scores of the individuals for later comparison
top_indivs = ga_pop.get_evaluated_individuals_from_hash(unique_indiv_hashlist)
top_indivs_train_scores = np.array([-indiv.stats[-1] for indiv in top_indivs])

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
chr_feat_coeffs = [indiv.coef_weights for indiv in keep_top_indivs]

for depth_int in range(n_interactions):

    depth_chr_feats = np.concatenate([chr[depth_int] for chr in chr_feats])
    depth_chr_coeffs = np.concatenate([chr[depth_int] for chr in chr_feat_coeffs])

    # Remove ones where equal to 0 entirely (or very close to 0)
    unused_feats = np.isclose(depth_chr_coeffs, 0.0)
    depth_chr_feats = depth_chr_feats[~unused_feats]
    depth_chr_coeffs = depth_chr_coeffs[~unused_feats]

    feat_used, feat_inv, feat_counts = np.unique(depth_chr_feats, return_inverse=True, return_counts=True, axis=0)

    feat_coef_sums = np.zeros(len(feat_used), dtype=np.float32)
    np.add.at(feat_coef_sums, feat_inv, depth_chr_coeffs)

    feat_names = np.array(col_ind_name_map)[feat_used]
    nominal_feat_prop = feat_counts / len(keep_top_indivs)
    len_feat_prop = feat_coef_sums / len(keep_top_indivs)
    sum_feat_prop = feat_coef_sums / feat_coef_sums.sum()

    critical_feat_ind = np.flatnonzero(feat_coef_sums > np.percentile(feat_coef_sums, critical_feat_percentile))
    critical_feat_ind = critical_feat_ind[np.argsort(feat_coef_sums[critical_feat_ind])]
    critical_feats_num = feat_used[critical_feat_ind]
    critical_feats = np.array(col_ind_name_map)[critical_feats_num]

    critical_len_feat_props = len_feat_prop[critical_feat_ind]
    critical_sum_feat_props = sum_feat_prop[critical_feat_ind]
    critical_nominal_feat_props = nominal_feat_prop[critical_feat_ind]

    for i, feat in enumerate(critical_feats_num):
        print(f"{feat} :: {critical_feats[i]} :: {critical_nominal_feat_props[i]:.4f} :: {critical_len_feat_props[i]:.4f} :: {critical_sum_feat_props[i]:.4f}")

    # Print output results to CSV
    out_data = pd.DataFrame({'Feature_nums': feat_used.tolist(), 'Feature_names': feat_names.tolist(), 'Nominal_usage_rate': nominal_feat_prop, 'Avg_coef_over_indiv': len_feat_prop, 'Avg_coef_over_sum_coef': sum_feat_prop, 'Total_feat_coef_sum': feat_coef_sums})
    out_data = out_data.sort_values(by=['Avg_coef_over_sum_coef'], ascending=False)
    csv_out_name = os.path.join(out_dir, f'feature_usage_depth{depth_int}.csv')
    out_data.to_csv(csv_out_name, index=False)

# Report (again)
print("\n")
print(f"Results output to: {out_dir}")
print(f"Number of features: {n_feats}")
print(f'Number of train/test samples: {len(y_train)} / {len(y_test)}')

ipdb.set_trace()