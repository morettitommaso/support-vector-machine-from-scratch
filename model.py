"""
Support Vector Machine from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - standardize_features
def standardize_features(x):
    x = x.astype(float)

    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)

    # evito divisione per zero
    std[std == 0] = 1

    return (x - mean) / std

# Step 2 - initialize_parameters
def initialize_parameters(n_features):
    """Return a dict with 'w' of shape (n_features,) and scalar 'b'."""
    # create starting weights and bias for a linear SVM
    
    w = np.zeros(n_features)
    b = 0.0

    return {'w': w, 'b': b}

# Step 3 - compute_scores
def compute_scores(x, params):
    """Return raw linear scores x @ w + b, shape (n_samples,)."""
    # score each example as a linear function of the current weights and bias.
    
    w = params["w"]
    b = params["b"]

    return ((x @ w) + b)

# Step 4 - predict_from_scores
def predict_from_scores(scores):
    # convert a 1-D array of raw scores into +1 / -1 class predictions.
    
    return np.where(scores >= 0, 1, -1)

# Step 5 - hinge_loss_example
def hinge_loss_example(score, y):
    # return the hinge loss for a single example with raw score `score` and label y in {-1, +1}.
    
    m = 1 - y * score # margin
    return float(max(0.0, m))

# Step 6 - svm_objective
def svm_objective(x, y, params, reg_lambda):
    # return mean hinge loss over the dataset plus reg_lambda * (w dot w)
    
    scores = compute_scores(x, params)
    losses = np.maximum(0, 1 - y * scores) # array di loss 

    mean_loss = np.mean(losses)
    reg = reg_lambda * np.sum(params["w"] ** 2)

    return float(mean_loss + reg)

# Step 7 - compute_gradients
def compute_gradients(x, y, params, reg_lambda):
    """Return {'dw': ndarray shape (n_features,), 'db': float} = gradient of svm_objective."""
    # compute the gradient of the SVM objective wrt params['w'] and params['b'].

    scores = compute_scores(x, params)
    margins = 1 - y * scores

    active = margins >= 0 # maschera per m >= 0

    dw = -np.sum(y[active, None] * x[active], axis=0) / len(x)
    dw += 2 * reg_lambda * params["w"]

    db = -np.sum(y[active]) / len(y)

    return {"dw": dw, "db": db}

# Step 8 - apply_update
def apply_update(params, grads, learning_rate):
    # return a new params dict after one gradient-descent step on 'w' and 'b'.
    
    return {
        "w": params["w"] - learning_rate * grads["dw"],
        "b": params["b"] - learning_rate * grads["db"],
    }

# Step 9 - train_svm
def train_svm(x, y, learning_rate, reg_lambda, n_epochs):
    # fit a linear SVM by repeatedly updating parameters over n_epochs passes.
    
    params = initialize_parameters(x.shape[1])

    for _ in range(n_epochs):

        grads = compute_gradients(x, y, params, reg_lambda)
        params = apply_update(params, grads, learning_rate)

    return params

# Step 10 - predict_labels
def predict_labels(x, params):
    # return an array of {-1, +1} labels, one per row of x, using params['w'] and params['b'].
    
    scores = compute_scores(x, params)
    return predict_from_scores(scores)

# Step 11 - accuracy_score
def accuracy_score(y_pred, y_true):
    # return the fraction of positions where y_pred equals y_true.

    return np.mean(y_pred == y_true)

