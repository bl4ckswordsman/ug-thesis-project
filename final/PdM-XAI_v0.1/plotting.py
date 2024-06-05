import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.colors import LogNorm
import os
import glob
from utils import ensure_dir

# Set the default font size
plt.rcParams['font.size'] = 14

# Set the font size for specific elements
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


def plot_metrics(metrics, model_name):
    metric_names = [name for name in metrics[0].keys() if name != 'confusion_matrix']

    fig, axs = plt.subplots(len(metric_names), 2, figsize=(12, 6 * len(metric_names)))

    for i, metric_name in enumerate(metric_names):
        metric_values = [metric[metric_name] for metric in metrics]
        axs[i, 0].boxplot(metric_values, vert=False)
        axs[i, 0].set_xlabel(metric_name.capitalize())
        axs[i, 0].set_title(f'{model_name} {metric_name.capitalize()} Across Folds')
        axs[i, 0].grid(True)

        axs[i, 1].hist(metric_values, bins=10)
        axs[i, 1].set_xlabel(metric_name.capitalize())
        axs[i, 1].set_title(f'{model_name} Histogram of {metric_name.capitalize()}s')
        axs[i, 1].grid(True)

    plt.tight_layout()
    plt_path = f'results/{model_name}/img/{model_name}_metrics_plots.png'
    ensure_dir(plt_path)
    plt.savefig(plt_path)


def plot_history(histories, folds, model_name):
    fig, axs = plt.subplots(len(histories), 2, figsize=(12, 12 * len(histories)))

    for i, history in enumerate(histories):
        axs[i, 0].plot(history['accuracy'], marker='o')
        axs[i, 0].set_title(f'{model_name} Model Accuracy for Fold {folds[i]}')
        axs[i, 0].set_ylabel('Accuracy')
        axs[i, 0].set_xlabel('Epoch')
        axs[i, 0].grid(True)

        axs[i, 1].plot(history['loss'], marker='o')
        axs[i, 1].set_title(f'{model_name} Model Loss for Fold {folds[i]}')
        axs[i, 1].set_ylabel('Loss')
        axs[i, 1].set_xlabel('Epoch')
        axs[i, 1].grid(True)

    plt.tight_layout()
    plt_path = f'results/{model_name}/img/history_fold_{folds[i]}.png'
    ensure_dir(plt_path)
    plt.savefig(plt_path)

    fig, axs = plt.subplots(2, 1, figsize=(12, 12))

    for i, history in enumerate(histories):
        axs[0].plot(history['accuracy'], label=f'Fold {folds[i]}', marker='o')
        axs[0].set_title(f'{model_name} Model Accuracy for All Folds')
        axs[0].set_ylabel('Accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].grid(True)

        axs[1].plot(history['loss'], label=f'Fold {folds[i]}', marker='o')
        axs[1].set_title(f'{model_name} Model Loss for All Folds')
        axs[1].set_ylabel('Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].grid(True)

    axs[0].legend()
    axs[1].legend()

    plt.tight_layout()
    all_plt_path = f'results/{model_name}/img/history_all_folds.png'
    ensure_dir(all_plt_path)
    plt.savefig(all_plt_path)


def plot_confusion_matrices(confusion_matrices, model_name, class_labels):
    fig, axs = plt.subplots(len(confusion_matrices), 1, figsize=(6, 6 * len(confusion_matrices)))

    # Define the logarithmic color scale
    log_norm = LogNorm(vmin=1, vmax=np.max(confusion_matrices))

    for i, conf_matrix in enumerate(confusion_matrices):
        # Create a heatmap from the confusion matrix with the logarithmic color scale
        sns.heatmap(conf_matrix, annot=True, fmt='d', ax=axs[i], cmap='Blues', xticklabels=class_labels,
                    yticklabels=class_labels, norm=log_norm)
        axs[i].set_title(f'{model_name} Confusion Matrix for Fold {i + 1}')
        axs[i].set_xlabel('Predicted Label')
        axs[i].set_ylabel('True Label')

    plt.tight_layout()

    plt_path = f'results/{model_name}/img/{model_name}_confusion_matrices.png'
    ensure_dir(plt_path)
    plt.savefig(plt_path)


def plot_model_metrics(model_dirs='results'):
    # Identify all the model directories in the `results` directory
    model_dirs = [d for d in os.listdir(model_dirs) if os.path.isdir(os.path.join(model_dirs, d))]

    # Initialize a dictionary to store metrics for all models
    all_metrics = {}

    # For each model directory, read the metrics CSV files
    for model_dir in model_dirs:
        csv_files = glob.glob(f'results/{model_dir}/csv/*_metrics_*.csv')

        # Initialize a dictionary to store metrics for the current model
        model_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': [], 'roc_auc': [], 'training_time': []}

        # Extract the required metrics from each CSV file
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            for metric in model_metrics.keys():
                model_metrics[metric].append(df[metric].mean())

        # Store the metrics for the current model in the all_metrics dictionary
        all_metrics[model_dir] = model_metrics

    # Plot the comparison of metrics across models
    for metric in ['accuracy', 'precision', 'recall', 'f1-score', 'roc_auc', 'training_time']:
        # Line chart
        plt.figure(figsize=(10, 6))
        for model, metrics in all_metrics.items():
            if metrics[metric]:  # Check if the list is not empty
                line, = plt.plot(metrics[metric], label=model)
                for i, value in enumerate(metrics[metric]):
                    plt.annotate(f'{value:.3f}', (i, value), textcoords="offset points", xytext=(0, 10),
                                 ha='center', color=line.get_color())
        plt.title(f'Comparison of {metric} across models')
        plt.xlabel('Fold')
        plt.ylabel(metric)
        plt.legend()

        # Save the line chart to the `results/all/img` directory
        plt_path = f'results/all/img/{metric}_comparison.png'
        ensure_dir(plt_path)
        plt.savefig(plt_path)
        plt.close()

        # Bar chart
        plt.figure(figsize=(10, 8))
        model_names = []
        mean_metrics = []
        std_metrics = []
        for model, metrics in all_metrics.items():
            if metrics[metric]:  # Check if the list is not empty
                model_names.append(model)
                mean_metrics.append(np.mean(metrics[metric]))
                std_metrics.append(np.std(metrics[metric]))

        # Plotting the bar chart
        bars = plt.bar(model_names, mean_metrics, yerr=std_metrics, capsize=10)
        plt.title(f'Comparison of {metric} across models')
        plt.xlabel('Model')
        plt.ylabel(metric)
        plt.xticks(rotation=47, ha='right')

        # Adjust the bottom margin
        plt.subplots_adjust(bottom=0.23)

        # Adding labels above the bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, round(yval, 3), ha='center', va='bottom')

        # Save the bar chart to the `results/all/img` directory
        plt_path = f'results/all/img/{metric}_comparison_bar.png'
        ensure_dir(plt_path)
        plt.savefig(plt_path)
        plt.close()

    # Convert the all_metrics dictionary to a DataFrame and save it as a CSV file
    all_metrics_df = pd.DataFrame(all_metrics)
    csv_path = 'results/all/csv/metrics_comparison.csv'
    ensure_dir(csv_path)
    all_metrics_df.to_csv(csv_path)


def plot_feature_importances(importances, std, indices, feature_names, model_name, fold):
    # Plot the feature importances
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(len(indices)), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation='vertical')
    plt.xlim([-1, len(indices)])

    # Adjust the bottom margin
    plt.subplots_adjust(bottom=0.40)

    # Save the plot
    plt_path = f'results/{model_name}/XAI_img/pfi_fold_{fold}.png'
    ensure_dir(plt_path)
    plt.savefig(plt_path)

