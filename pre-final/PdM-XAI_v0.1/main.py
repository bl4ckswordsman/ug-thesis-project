import joblib
import pandas as pd
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical

from evaluation import evaluate_and_append_accuracy, plot_metrics, plot_history, plot_confusion_matrices
from model import create_and_train_model
from preprocessing import load_and_preprocess_data

# Load and preprocess the data
X, y, le = load_and_preprocess_data()

# Initialize the cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store metrics
metrics = []

# Directory to store the model data
model_dir = 'model_data'

# Initialize model and model_name to None
model = None
model_name = None

# Cross-validation
for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Convert labels to categorical one-hot encoding
    y_train_categorical = to_categorical(y_train, num_classes=len(le.classes_))
    y_test_categorical = to_categorical(y_test, num_classes=len(le.classes_))

    # Create and train the model
    model, model_name = create_and_train_model(X_train, y_train_categorical, len(le.classes_), fold)

    # Evaluate the model and append the accuracy to the list
    evaluate_and_append_accuracy(model, model_name, X_test, y_test_categorical, metrics, fold)

# Load the histories from csv
histories = [pd.read_csv(f'results/{model_name}/csv/history_fold_{fold}.csv') for fold in range(1, kf.get_n_splits() + 1)]
folds = list(range(1, kf.get_n_splits() + 1))

# Plot the histories
plot_history(histories, folds, model_name)

# Plot the metrics
plot_metrics(metrics, model_name)

# Extract confusion matrices from accuracies
confusion_matrices = [metrics['confusion_matrix'] for metrics in metrics]

# Plot the confusion matrices
plot_confusion_matrices(confusion_matrices, model_name)

# Save the model
model.save(model_dir + '/' + model_name + '/model.keras')

# Save the label encoder
joblib.dump(le, (model_dir + '/' + model_name + '/label_encoder.joblib'))
