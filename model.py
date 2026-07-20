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

# Step 2 - initialize_parameters (not yet solved)
# TODO: implement

# Step 3 - compute_scores (not yet solved)
# TODO: implement

# Step 4 - predict_from_scores (not yet solved)
# TODO: implement

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

