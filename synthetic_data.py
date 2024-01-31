import numpy as np

from sklearn.utils import check_random_state

def make_binary_classification_with_interaction(
        n_samples=100,
        n_features=1000,
        n_informative=10,
        n_interactions=10,
        n_interactions_from_informative=5,
        flip_y=0.01,
        random_state=None,
):
    # Heavily inspired by sklearn's make_classification
    if n_interactions is None:
        n_interactions = int(n_informative / 2)
    
    if n_interactions_from_informative is None:
        n_interactions_from_informative = n_interactions

    assert n_interactions_from_informative < n_informative, 'Cannot have more interactions from informative than total interactions'
    
    generator = check_random_state(random_state)

    # Initialize X and y
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)

    # Initially draw informative features from the standard normal
    X[:, :n_informative] = generator.standard_normal(size=(n_samples, n_informative))

    # Fill useless (single-order) features with random numbers
    n_useless = n_features - n_informative - n_interactions
    X[:, -n_useless:] = generator.standard_normal(size=(n_samples, n_useless))

    # Select some subset of the informative features for interactions
    info_interaction_features = generator.choice(n_informative, (n_interactions_from_informative,2), replace=False)

    # Select some subset of the useless features for interactions
    n_interactions_from_useless = n_interactions - n_interactions_from_informative
    noninfo_interaction_features = generator.choice(n_useless, (n_interactions_from_useless,2), replace=False) + n_informative + n_interactions

    all_interaction_features = np.vstack((info_interaction_features, noninfo_interaction_features))

    for i in range(n_interactions):
        X[:, n_informative+i+1] = X[:, all_interaction_features[i,0]] * X[:, all_interaction_features[i,1]]
    
    # Set y based on the sum of the informative features
    total_info_feat = n_interactions + n_informative
    X_k = X[:, :total_info_feat]  # slice a view of the informative features
    y = (X_k.sum(axis=1) > 0).astype(int)

    # Randomly replace labels
    if flip_y >= 0.0:
        flip_mask = generator.uniform(size=n_samples) < flip_y
        y[flip_mask] = generator.randint(2, size=flip_mask.sum())

    # Return the feature interaction columns to randomness
    # (since they shouldn't be in the actual dataset)
    for i in range(n_interactions):
        X[:, n_informative+i+1] = generator.standard_normal(size=(n_samples,))

    # Sort interaction features along rows and then by first column
    all_interaction_features = np.sort(all_interaction_features, axis=1)
    all_interaction_features = all_interaction_features[np.argsort(all_interaction_features[:,0])]

    return X, y, all_interaction_features


# For seed = 999:
# [[  8  14]
#  [  9  42]
#  [ 18  49]
#  [ 26  36]
#  [ 40  43]
#  [113 226]
#  [190 969]
#  [348 675]
#  [552 951]
#  [797 898]]