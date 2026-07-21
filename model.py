"""
Support Vector Machine from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - standardize_features
import numpy as np

def standardize_features(x):
    x = x.astype(float)

    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)

    # evito divisione per zero
    std[std == 0] = 1

    return (x - mean) / std

# Step 2 - initialize_parameters
import numpy as np

def initialize_parameters(n_features):
    """Return a dict with 'w' of shape (n_features,) and scalar 'b'."""
    # TODO: create starting weights and bias for a linear SVM
    
    w = np.zeros(n_features)
    b = 0.0

    return {'w': w, 'b': b}

# Step 3 - compute_scores
import numpy as np

def compute_scores(x, params):
    """Return raw linear scores x @ w + b, shape (n_samples,)."""
    # TODO: score each example as a linear function of the current weights and bias.
    
    w = params["w"]
    b = params["b"]

    return ((x @ w) + b)

# Step 4 - predict_from_scores
import numpy as np

def predict_from_scores(scores):
    # TODO: convert a 1-D array of raw scores into +1 / -1 class predictions.
    
    predictions = []

    for s in scores:

        if s >= 0:
            predictions.append(1)
            continue
        predictions.append(-1)

    return np.array(predictions)

# Step 5 - hinge_loss_example (not yet solved)
# TODO: implement

# Step 6 - svm_objective (not yet solved)
# TODO: implement

# Step 7 - compute_gradients (not yet solved)
# TODO: implement

# Step 8 - apply_update (not yet solved)
# TODO: implement

# Step 9 - train_svm (not yet solved)
# TODO: implement

# Step 10 - predict_labels (not yet solved)
# TODO: implement

# Step 11 - accuracy_score (not yet solved)
# TODO: implement

