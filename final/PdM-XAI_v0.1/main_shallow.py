import joblib
import pandas as pd
from sklearn.model_selection import KFold

from evaluation import evaluate_and_append_accuracy_tree
from plotting import plot_metrics, plot_history, plot_confusion_matrices, plot_model_metrics
from preprocessing import load_and_preprocess_data
from tree_models import create_and_train_tree_model, create_tree_model1, create_tree_model2
from utils import ensure_dir
from explainabilty import explain_model_with_pfi, keras_score, sklearn_score

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

    # Create and train the model
    model, model_name = create_and_train_tree_model(X_train, y_train,
                                                    create_tree_model2, fold)

    # Save the test data
    test_data_path = f'{model_dir}/{model_name}/test_data_fold_{fold}.joblib'
    ensure_dir(test_data_path)
    joblib.dump({'X_test': X_test, 'y_test': y_test}, test_data_path)

    # Load the test data
    test_data = joblib.load(test_data_path)
    X_test, y_test = test_data['X_test'], test_data['y_test']

    # Evaluate the model and append the accuracy to the list
    evaluate_and_append_accuracy_tree(model, model_name, X_test, y_test, metrics, fold)

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

# Save the model (TREE MODELS)
model_path = f'{model_dir}/{model_name}/model.joblib'
ensure_dir(model_path)
joblib.dump(model, model_path)

# Save the label encoder
le_path = f'{model_dir}/{model_name}/label_encoder.joblib'
ensure_dir(le_path)
joblib.dump(le, le_path)