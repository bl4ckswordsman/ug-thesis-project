import numpy as np
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import get_scorer

from plotting import plot_feature_importances
from utils import ensure_dir


def keras_score(estimator, X, y):
    loss, accuracy = estimator.evaluate(X, y)
    return accuracy


def calculate_pfi(model, x_val, y_val, scoring):
    # Calculate permutation importances
    result = permutation_importance(model, x_val, y_val, scoring=scoring, n_repeats=10, random_state=42)

    # Get importances and their standard deviations
    importances = result.importances_mean
    std = result.importances_std

    # Get the indices of the importances sorted in decreasing order
    indices = np.argsort(importances)[::-1]

    return importances, std, indices


def explain_model_with_pfi(model, model_name, x_val, y_val, fold, scoring, feature_names, all_importances=None, all_std=None):
    if model is not None:
        importances, std, indices = calculate_pfi(model, x_val, y_val, scoring)
        plot_feature_importances(importances, std, indices, feature_names, model_name, fold)

        # If all_importances and all_std are provided, append the current importances and std
        if all_importances is not None and all_std is not None:
            all_importances.append(importances)
            all_std.append(std)

    # If fold is None, it means we want to plot the combined feature importances
    if fold is None and all_importances is not None and all_std is not None:
        # Calculate mean importances and standard deviations across all folds
        mean_importances = np.mean(all_importances, axis=0)
        mean_std = np.mean(all_std, axis=0)

        # Get the indices of the importances sorted in decreasing order
        indices = np.argsort(mean_importances)[::-1]

        plot_feature_importances(mean_importances, mean_std, indices, feature_names, model_name, 'combined')

        # Reset importances, std, and indices to None
        importances = std = indices = None

    # Return importances, standard deviations, and indices
    return importances, std, indices
