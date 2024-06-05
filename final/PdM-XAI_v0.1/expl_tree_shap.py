import numpy as np
import scipy as sp
import shap
from matplotlib import pyplot as plt
import matplotlib

from utils import ensure_dir

# Set the default font size
plt.rcParams['font.size'] = 16

# Set the font size for specific elements
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14

# Use a non-interactive backend suitable for script execution
matplotlib.use('Agg')


def compute_hierarchical_clustering(X, model_name, save_path=None):
    partition_tree = shap.utils.partition_tree(X)
    plt.figure(figsize=(10, 7))
    sp.cluster.hierarchy.dendrogram(partition_tree, labels=X.columns)
    plt.title(f"{model_name}\n Hierarchical Clustering Dendrogram")
    plt.xlabel("feature")
    plt.ylabel("distance")
    plt.xticks(rotation=47, ha='right')
    plt.subplots_adjust(bottom=0.3)
    if save_path:
        ensure_dir(save_path)
        plt.savefig(save_path)
    plt.show()
    return partition_tree


def explain_instance(model, instance):
    # Create a SHAP explainer using the Tree method
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(instance)

    return shap_values

def plot_additional_shap_plots(model, X, shap_values, feature_names, instance, class_labels, output_dir):
    # Ensure output directory exists
    ensure_dir(output_dir)

    # Aggregate SHAP values per class
    class_shap_values = {class_label: [] for class_label in class_labels}

    for i in range(len(shap_values)):
        for j in range(len(shap_values[i])):
            single_shap_value = np.array(shap_values[i][j])
            for k, class_label in enumerate(class_labels):  # Iterate through each class
                class_shap_value = single_shap_value[:, k]  # Select SHAP values for the specific class
                if len(class_shap_value) == len(feature_names):
                    class_shap_values[class_label].append(class_shap_value)
                else:
                    print(
                        f"Feature name length ({len(feature_names)}) and SHAP values length ({len(class_shap_value)}) mismatch for class {class_label}")

    # Combine and plot SHAP values for each class
    for class_label, values in class_shap_values.items():
        if values:
            combined_shap_values = np.mean(values, axis=0)
            combined_shap_values = -combined_shap_values  # Invert the SHAP values
            plt.figure(figsize=(10, 7))  # Create a new figure with increased width
            explanation = shap.Explanation(
                values=combined_shap_values,
                base_values=np.mean([model.predict_proba(instance)[0][k] for k in range(len(class_labels))]),
                data=instance.iloc[0],
                feature_names=feature_names
            )
            shap.waterfall_plot(explanation)
            plt.title(f"Waterfall Plot - Class: {class_label}")
            waterf_path = f'{output_dir}/waterfall_plot_class_{class_label}.png'
            ensure_dir(waterf_path)
            plt.savefig(waterf_path, bbox_inches='tight')  # Use bbox_inches to include all content
            plt.close()
