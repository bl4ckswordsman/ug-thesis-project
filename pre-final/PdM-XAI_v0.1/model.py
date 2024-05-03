from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pandas as pd

epochs_nr = 50
batch_size_nr = 10


def create_and_train_model(x_train, y_train, num_classes, fold):
    # Define the model inside the function
    model = Sequential()
    model.add(Input(shape=(x_train.shape[1],)))  # Input layer
    model.add(Dense(32, activation='relu'))  # Hidden layer 1, 32 neurons, ReLU(Rectified Linear Unit) activation
    model.add(Dense(64, activation='relu'))  # Hidden layer 2
    model.add(Dense(num_classes, activation='softmax'))  # Output layer

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model and save the history
    history = model.fit(x_train, y_train, epochs=epochs_nr, batch_size=batch_size_nr)

    # Convert the history.history dict to a pandas DataFrame
    hist_df = pd.DataFrame(history.history)

    # Save to csv
    model_name = type(model).__name__
    hist_df.to_csv(f'results/{model_name}/history_fold_{fold}.csv')

    return model, model_name
