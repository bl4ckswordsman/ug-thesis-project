import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from utils import ensure_dir


def evaluate_and_append_accuracy(model, model_name, x_test, y_test, accuracies, fold):
    y_score = model.predict(x_test)
    y_pred = np.argmax(y_score, axis=-1)
    y_test = np.argmax(y_test, axis=-1)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_score, multi_class='ovo')
    conf_matrix = confusion_matrix(y_test, y_pred)

    metrics = {
        'accuracy': report['accuracy'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1-score': report['weighted avg']['f1-score'],
        'roc_auc': roc_auc,
        'confusion_matrix': conf_matrix
    }

    accuracies.append(metrics)

    metrics_df = pd.DataFrame(accuracies)
    metrics_path = f'results/{model_name}/csv/{model_name}_metrics_fold_{fold}.csv'
    ensure_dir(metrics_path)
    metrics_df.to_csv(metrics_path, index=False)

    # print(metrics_df)


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
        sns.heatmap(conf_matrix, annot=True, fmt='d', ax=axs[i], cmap='Blues', xticklabels=class_labels, yticklabels=class_labels, norm=log_norm)
        axs[i].set_title(f'{model_name} Confusion Matrix for Fold {i + 1}')
        axs[i].set_xlabel('Predicted Label')
        axs[i].set_ylabel('True Label')

    plt.tight_layout()

    plt_path = f'results/{model_name}/img/{model_name}_confusion_matrices.png'
    ensure_dir(plt_path)
    plt.savefig(plt_path)
