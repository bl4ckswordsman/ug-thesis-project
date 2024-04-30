import matplotlib.pyplot as plt
import pandas as pd


def evaluate_and_append_accuracy(model, x_test, y_test, accuracies, fold):
    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test)
    accuracies.append(accuracy)

    # Print the average accuracy
    print(f'Accuracy in fold {fold}:', accuracy)


def plot_accuracies(accuracies):
    # Box plot of accuracies
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.boxplot(accuracies, vert=False)
    plt.xlabel('Accuracy')
    plt.title('Model Accuracy Across Folds')

    # Histogram of accuracies
    plt.subplot(1, 2, 2)
    plt.hist(accuracies, bins=10)
    plt.xlabel('Accuracy')
    plt.title('Histogram of Model Accuracies')

    plt.savefig('results/accuracy_plots.png')


def plot_history(fold):
    # Load the history from csv
    history = pd.read_csv(f'results/history_fold_{fold}.csv')

    # Line plot of accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')

    # Line plot of loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.savefig(f'results/history_fold_{fold}.png')
