import joblib
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical

from evaluation import evaluate_and_append_accuracy, plot_accuracies, plot_history
from model import create_and_train_model
from preprocessing import load_and_preprocess_data

# Load and preprocess the data
X, y, le = load_and_preprocess_data()

# Initialize the cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize lists to store metrics
accuracies = []

# Directory to store the model data
model_dir = 'model_data'

# Cross-validation
for fold, (train_index, test_index) in enumerate(kf.split(X), 1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Convert labels to categorical one-hot encoding
    y_train_categorical = to_categorical(y_train, num_classes=len(le.classes_))
    y_test_categorical = to_categorical(y_test, num_classes=len(le.classes_))

    # Create and train the model
    model = create_and_train_model(X_train, y_train_categorical, len(le.classes_), fold)

    # Evaluate the model and append the accuracy to the list
    evaluate_and_append_accuracy(model, X_test, y_test_categorical, accuracies, fold)

    # Plot the history
    plot_history(fold)

# Plot the accuracy
plot_accuracies(accuracies)

# Save the model
model.save(model_dir + '/model.keras')

# Save the label encoder
joblib.dump(le, (model_dir + 'label_encoder.joblib'))
