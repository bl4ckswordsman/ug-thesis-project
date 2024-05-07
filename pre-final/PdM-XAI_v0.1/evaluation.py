import numpy as np
import pandas as pd
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

