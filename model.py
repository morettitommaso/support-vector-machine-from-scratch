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
    
    return np.where(scores >= 0, 1, -1)

# Step 5 - hinge_loss_example
def hinge_loss_example(score, y):
    # TODO: return the hinge loss for a single example with raw score `score` and label y in {-1, +1}.
    
    m = 1 - y * score # margin
    return float(max(0.0, m))

# Step 6 - svm_objective
def svm_objective(x, y, params, reg_lambda):
    # TODO: return mean hinge loss over the dataset plus reg_lambda * (w dot w)
    
    scores = compute_scores(x, params)
    losses = np.maximum(0, 1 - y * scores) # array di loss 

    mean_loss = np.mean(losses)
    reg = reg_lambda * np.sum(params["w"] ** 2)

    return float(mean_loss + reg)

# Step 7 - compute_gradients
import numpy as np

def compute_gradients(x, y, params, reg_lambda):
    """Return {'dw': ndarray shape (n_features,), 'db': float} = gradient of svm_objective."""
    # TODO: compute the gradient of the SVM objective wrt params['w'] and params['b'].

    scores = compute_scores(x, params)
    margins = 1 - y * scores

    active = margins >= 0 # maschera per m >= 0

    dw = -np.sum(y[active, None] * x[active], axis=0) / len(x)
    dw += 2 * reg_lambda * params["w"]

    db = -np.sum(y[active]) / len(y)

    return {"dw": dw, "db": db}

# Step 8 - apply_update (not yet solved)
# TODO: implement

# Step 9 - train_svm (not yet solved)
# TODO: implement

# Step 10 - predict_labels (not yet solved)
# TODO: implement

# Step 11 - accuracy_score (not yet solved)
# TODO: implement

