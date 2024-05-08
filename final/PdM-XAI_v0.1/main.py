import joblib
import pandas as pd
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical

from evaluation import evaluate_and_append_accuracy
from plotting import plot_metrics, plot_history, plot_confusion_matrices, plot_model_metrics
from model import create_and_train_model, create_seq_model, create_deep_model, create_deep_model2
from preprocessing import load_and_preprocess_data
from utils import ensure_dir
from explainabilty import explain_model_with_pfi, keras_score

# Load and preprocess the data
X, y, le = load_and_preprocess_data()

# Get the class labels
class_labels = le.classes_

# Initialize the cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store metrics
metrics = []

# Directory to store the model data
model_dir = 'model_data'

# Initialize model and model_name to None
model = None
model_name = None

# Initialize lists to store importances and standard deviations for each fold
all_importances = []
all_std = []

# Cross-validation
for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Convert labels to categorical one-hot encoding
    y_train_categorical = to_categorical(y_train, num_classes=len(le.classes_))
    y_test_categorical = to_categorical(y_test, num_classes=len(le.classes_))

    # Create and train the model
    model, model_name = create_and_train_model(X_train, y_train_categorical,
                                               create_deep_model2, len(le.classes_), fold)

    # Save the test data
    test_data_path = f'{model_dir}/{model_name}/test_data_fold_{fold}.joblib'
    ensure_dir(test_data_path)
    joblib.dump((X_test, y_test_categorical), test_data_path)

    # Explain the model with Permutation Feature Importance
    importances, std, indices = explain_model_with_pfi(model, model_name,
                                                       X_test, y_test_categorical, fold, scoring=keras_score,
                                                       feature_names=X.columns, all_importances=all_importances,
                                                       all_std=all_std)

    # Evaluate the model and append the accuracy to the list
    evaluate_and_append_accuracy(model, model_name, X_test, y_test_categorical, metrics, fold)

# After the cross-validation loop, generate the combined plot
explain_model_with_pfi(None, model_name, X, None, None, None, X.columns, all_importances=all_importances,
                       all_std=all_std)

# Load the histories from csv
histories = [pd.read_csv(
    f'results/{model_name}/csv/history_fold_{fold}.csv') for fold in range(1, kf.get_n_splits() + 1)]
folds = list(range(1, kf.get_n_splits() + 1))

# Plot the histories
plot_history(histories, folds, model_name)

# Plot the metrics
plot_metrics(metrics, model_name)

# Extract confusion matrices from accuracies
confusion_matrices = [metrics['confusion_matrix'] for metrics in metrics]

# Plot the confusion matrices
plot_confusion_matrices(confusion_matrices, model_name, class_labels)

# Plot the comparison of models
plot_model_metrics()

# Save the model
model_path = f'{model_dir}/{model_name}/model.keras'
ensure_dir(model_path)
model.save(model_path)

# Save the label encoder
le_path = f'{model_dir}/{model_name}/label_encoder.joblib'
ensure_dir(le_path)
joblib.dump(le, le_path)
