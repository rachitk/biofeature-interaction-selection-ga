import numpy as np

from .utils import softmax
from .individual import Individual
from .chromosome import Chromosome

import ipdb

# File that defines mutations that can be used by the population
# All functions are expected to take in an Individual and return an Individual
# with population metadata being passed with info about the population,
# with an RNG being passed for fixing randomness, and
# with an optional scaling weight for each feature being passed (TODO)


def add_feature(individual, pop_metadata, rng_seed=None, scale_weight=None):
    # Add a feature to an individual
    # Selection is based on a uniform distribution (or weights)
    # for the chromosome that is selected
    # with an additional scaling if chromosome depth > 1
    rng = np.random.default_rng(rng_seed)

    # Select a random chromosome (TODO: maybe scale by length of chromosome)
    sel_chr_num = rng.choice(pop_metadata['interaction_num'])
    chr_others = [i for i in range(pop_metadata['interaction_num']) if i != sel_chr_num]
    sel_chr = individual.chromosomes[sel_chr_num]

    # Scale addition based on whether a feature is present in the other chromosomes
    # based entirely on presence only - if present in other chromosome, 
    # the feature is more likely to be added compared to a uniform baseline
    # based on how many other chromosomes it is present in
    feat_scale = np.zeros(shape=(pop_metadata['num_features']))

    for chr_num in chr_others:
        feat_scale[np.unique(individual.chromosomes[chr_num].features)] += 1.0

    feat_add = rng.choice(pop_metadata['num_features'], size=(1, sel_chr.depth), p=softmax(feat_scale))
    new_chr = Chromosome(np.append(sel_chr.features, feat_add, axis=0))

    return Individual([individual.chromosomes[chr_num] if chr_num != sel_chr_num else new_chr 
                       for chr_num in range(pop_metadata['interaction_num'])])


def remove_feature(individual, pop_metadata, rng_seed=None, scale_weight=None):
    # Remove a feature from an individual
    # Will remove based on coefficients in ElasticNet
    # for the chromosome that is selected
    rng = np.random.default_rng(rng_seed)

    # Prevent selecting empty chromosomes
    # and compute for each chromosome the coefficient proportions
    # More likely to select a chromosome with a lower coefficient sum
    nonempty_chrs = [i for i in range(pop_metadata['interaction_num']) if len(individual.chromosomes[i]) > 0]
    nonempty_selprops = [1-individual.coef_weights[i].sum() for i in nonempty_chrs]

    # Select a random chromosome
    sel_chr_num = rng.choice(nonempty_chrs, p=softmax(np.array(nonempty_selprops)))
    chr_others = [i for i in range(pop_metadata['interaction_num']) if i != sel_chr_num]
    sel_chr = individual.chromosomes[sel_chr_num]

    # Scale removal based on whether a feature is present in the other chromosomes
    # based entirely on presence only - if present in other chromosome, 
    # the feature is more likely to be removed compared to a uniform baseline
    # based on the proportion of the other chromosomes it is present in
    feat_scale = np.zeros(shape=(pop_metadata['num_features'])) - np.inf
    feat_scale[np.unique(sel_chr.features)] = 0.0

    for chr_num in chr_others:
        feat_scale[np.unique(individual.chromosomes[chr_num].features)] += 1.0 / pop_metadata['interaction_num']

    # Further scale removal based on feature coefficients from the model used
    # to fit the data (from the inverse of individual.coef_weights, scaled by the number of
    # features present in the chromosome to offset probabilistic scaling)
    feature_coef_scales = softmax(1 / individual.coef_weights[sel_chr_num]) * len(individual.coef_weights[sel_chr_num])
    feature_coef_scales = np.tile(np.expand_dims(feature_coef_scales, axis=1), (1, sel_chr.depth))
    feat_scale[sel_chr.features.flatten()] += feature_coef_scales.flatten()

    # Actually select the feature number to be removed
    feat_remove = rng.choice(pop_metadata['num_features'], p=softmax(feat_scale))
    
    # Get all elements in this chromosome that contain the feature
    # to be removed, and then pick one to remove randomly (uniform)
    feat_rows, _ = np.where(sel_chr.features == feat_remove)
    remove_feat_mask = np.ones(len(sel_chr.features), dtype=bool)
    remove_feat_mask[rng.choice(feat_rows)] = False
    new_chr = Chromosome(sel_chr.features[remove_feat_mask])

    return Individual([individual.chromosomes[chr_num] if chr_num != sel_chr_num else new_chr 
                       for chr_num in range(pop_metadata['interaction_num'])])


def alter_feature_depth(individual, pop_metadata, rng_seed=None, scale_weight=None):
    # Convert a feature to an interaction or vice-versa
    rng = np.random.default_rng(rng_seed)

    # Prevent selecting empty chromosomes
    nonempty_chrs = [i for i in range(pop_metadata['interaction_num']) if len(individual.chromosomes[i]) > 0]

    # If there are less than two chromosomes to select, then instead
    # do an add/remove mutation with equal probability
    if(len(nonempty_chrs) < 2):
        alt_mutate = rng.choice([add_feature, remove_feature])
        return alt_mutate(individual, pop_metadata, rng_seed, scale_weight)

    # Select two random chromosomes (TODO: maybe scale by length of chromosome or prevent selecting empty?)
    sel_chr_nums = rng.choice(nonempty_chrs, size=(2,), replace=False)
    src_chr = individual.chromosomes[sel_chr_nums[0]]
    dst_chr = individual.chromosomes[sel_chr_nums[1]]
    chr_others = [i for i in range(pop_metadata['interaction_num']) if i not in sel_chr_nums]

    # Be more likely to select a feature for depth alteration
    # if it had a low coefficient in the original dataset relative 
    # to other features of the same depth
    src_feat_scale = softmax(individual.coef_weights[sel_chr_nums[0]])
    src_feat_sel_ind = rng.choice(len(src_feat_scale), p=src_feat_scale)
    src_feat_sel = src_chr.features[src_feat_sel_ind]

    src_chr_new = Chromosome(np.delete(src_chr.features, src_feat_sel_ind, axis=0))

    # Handle based on promotion or demotion situation
    if(src_chr.depth > dst_chr.depth):
        # if this is an interaction becoming a single feature
        # (or becoming a shallower interaction)
        # then just select the appropriate number and add it
        dst_feat_new = rng.choice(src_feat_sel, size=(1,dst_chr.depth))
    else:
        # if this is a single feature becoming an interaction
        # then pick a random feature uniformly and add it to the set
        addl_feats = rng.choice(pop_metadata['num_features'], size=(dst_chr.depth - src_chr.depth,))
        dst_feat_new = np.expand_dims(np.concatenate([src_feat_sel, addl_feats]), axis=0)

    dst_chr_new = Chromosome(np.concatenate([dst_chr.features, dst_feat_new], axis=0))
        
    return Individual([src_chr_new if chr_num == sel_chr_nums[0] else 
                       dst_chr_new if chr_num == sel_chr_nums[1] else 
                       individual.chromosomes[chr_num] 
                       for chr_num in range(pop_metadata['interaction_num'])])
    