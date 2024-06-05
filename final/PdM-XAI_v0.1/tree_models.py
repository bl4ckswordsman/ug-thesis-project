import time

import pandas as pd
from sklearn.metrics import accuracy_score, zero_one_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from resource_logger import log_cpu_usage
from utils import ensure_dir


def create_tree_model1(max_depth=None):
    model = DecisionTreeClassifier(max_depth=max_depth)
    return model, "DecisionTree"


def create_tree_model2(n_estimators=100, max_depth=None):
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    return model, "RandomForest"


def create_and_train_tree_model(x_train, y_train, create_model_func, fold):
    # Create the model using the provided function
    model, model_name = create_model_func()

    # Record the start time and CPU usage
    start_time = time.time()
    start_cpu = log_cpu_usage()

    # Train the model
    model.fit(x_train, y_train)

    # Record the end time and CPU usage
    end_time = time.time()
    end_cpu = log_cpu_usage()

    # Calculate the training time and CPU usage
    training_time = end_time - start_time
    cpu_used = end_cpu - start_cpu

    # Calculate the accuracy
    y_pred = model.predict(x_train)
    accuracy = accuracy_score(y_train, y_pred)

    # Calculate the misclassification rate
    misclassification_rate = zero_one_loss(y_train, y_pred)

    # Save the training time, accuracy, misclassification rate, and CPU usage
    history_df = pd.DataFrame({'training_time': [training_time], 'accuracy': [accuracy], 'loss': [misclassification_rate], 'cpu_used': [cpu_used]})

    # Save to csv
    hist_path = f'results/{model_name}/csv/history_fold_{fold}.csv'
    ensure_dir(hist_path)
    history_df.to_csv(hist_path)

    return model, model_name
