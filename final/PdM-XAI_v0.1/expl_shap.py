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


def explain_instance(model, instance, X):
    # Ensure the instance and X are float32 for SHAP
    instance = instance.astype('float32')
    X = X.astype('float32')

    # Summarize the background data using a sample
    X_summary = shap.sample(X, 100)

    # Create a SHAP explainer using the Kernel method
    explainer = shap.KernelExplainer(model.predict, X_summary)
    shap_values = explainer.shap_values(instance)

    return shap_values


def aggregate_shap_values(shap_values):
    # Convert the SHAP values to a numpy array
    shap_values = np.squeeze(np.array(shap_values))
    # If the SHAP values are 3D, average the absolute values across the last dimension
    if len(shap_values.shape) == 3:
        aggregated_shap_values = np.mean(np.abs(shap_values), axis=2)
    else:
        aggregated_shap_values = shap_values
    return aggregated_shap_values


def plot_shap_values(shap_values, feature_names, class_labels, save_path=None):
    # Aggregate the SHAP values
    shap_values = aggregate_shap_values(shap_values)
    # Check if the number of features matches the number of SHAP values
    if shap_values.shape[1] == len(feature_names):
        shap.summary_plot(shap_values, feature_names=feature_names, plot_type="violin")
        if save_path:
            ensure_dir(save_path)
            plt.savefig(save_path)
        plt.show()
    else:
        print(f"Feature name length ({len(feature_names)}) and SHAP values length ({shap_values.shape[1]}) mismatch")


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
                base_values=np.mean([model.predict(instance)[0][k] for k in range(len(class_labels))]),
                data=instance.iloc[0],
                feature_names=feature_names
            )
            shap.waterfall_plot(explanation)
            plt.title(f"Waterfall Plot - Class: {class_label}")
            waterf_path = f'{output_dir}/waterfall_plot_class_{class_label}.png'
            ensure_dir(waterf_path)
            plt.savefig(waterf_path, bbox_inches='tight')  # Use bbox_inches to include all content
            plt.close()

    # Force plot for each output (optional, similar approach can be followed)
    # for class_label, values in class_shap_values.items():
    #     if values:
    #         combined_shap_values = np.mean(values, axis=0)
    #         shap.force_plot(
    #             base_value=np.mean([model.predict(instance)[0][k] for k in range(len(class_labels))]),
    #             shap_values=combined_shap_values,
    #             features=instance.iloc[0],
    #             feature_names=feature_names,
    #             matplotlib=True
    #         )
    #         plt.title(f"Force Plot - Class: {class_label}")
    #         force_path = f'{output_dir}/force_plot_class_{class_label}.png'
    #         ensure_dir(force_path)
    #         plt.savefig(force_path)
    #         plt.close()

    # Dependence plot and Decision plot can be similarly combined if needed
