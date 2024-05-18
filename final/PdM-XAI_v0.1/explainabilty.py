import numpy as np
from sklearn.inspection import permutation_importance
from lime import lime_tabular
from sklearn.metrics import accuracy_score

from plotting import plot_feature_importances


def keras_score(estimator, X, y):
    loss, accuracy = estimator.evaluate(X, y)
    return accuracy


def sklearn_score(estimator, X, y):
    y_pred = estimator.predict(X)
    return accuracy_score(y, y_pred)


def calculate_pfi(model, x_val, y_val, scoring):
    # Calculate permutation importances
    result = permutation_importance(model, x_val, y_val, scoring=scoring, n_repeats=10, random_state=42)

    # Get importances and their standard deviations
    importances = result.importances_mean
    std = result.importances_std

    # Get the indices of the importances sorted in decreasing order
    indices = np.argsort(importances)[::-1]

    return importances, std, indices


def explain_model_with_pfi(model, model_name, x_val, y_val, fold, scoring,
                           feature_names, all_importances=None, all_std=None):
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


def explain_model_with_lime(model, model_name, x_val, y_val, fold, feature_names, all_explanations=None):
    # Create a LimeTabularExplainer
    explainer = lime_tabular.LimeTabularExplainer(x_val, feature_names=feature_names, class_names=['0', '1'],
                                                  verbose=True, mode='classification')

    if model is not None:
        # Explain a prediction
        explanation = explainer.explain_instance(x_val[0], model.predict_proba, num_features=5)

        # If all_explanations is provided, append the current explanation
        if all_explanations is not None:
            all_explanations.append(explanation)

    # If fold is None, it means we want to plot the combined explanations
    if fold is None and all_explanations is not None:
        # Calculate mean explanations across all folds
        mean_explanations = np.mean(all_explanations, axis=0)

        # Plot the explanations
        plot_feature_importances(mean_explanations, feature_names, model_name, 'combined')

        # Reset explanations to None
        explanations = None

    # Return explanations
    return explanations
