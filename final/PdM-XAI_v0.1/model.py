from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
import pandas as pd
from tensorflow.keras.regularizers import l2

from utils import ensure_dir

epochs_nr = 50
batch_size_nr = 10


def create_seq_model(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))  # Input layer
    model.add(Dense(32, activation='relu'))  # Hidden layer 1, 32 neurons, ReLU(Rectified Linear Unit) activation
    model.add(Dense(64, activation='relu'))  # Hidden layer 2
    model.add(Dense(num_classes, activation='softmax'))  # Output layer

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model, "Seq_2Layer_32_64_ReLU_Adam"


def create_deep_model(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))  # Input layer
    model.add(Dense(64, activation='relu'))  # Hidden layer 1, 64 neurons, ReLU(Rectified Linear Unit) activation
    model.add(Dropout(0.5))  # Dropout layer 1
    model.add(Dense(128, activation='relu'))  # Hidden layer 2
    model.add(Dropout(0.5))  # Dropout layer 2
    model.add(Dense(64, activation='relu'))  # Hidden layer 3
    model.add(Dense(num_classes, activation='softmax'))  # Output layer

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model, "Deep_3Layer_64_128_64_ReLU_Adam"


def create_deep_model2(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))  # Input layer
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))  # Hidden layer 1, 128 neurons, ReLU activation
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # Dropout layer 1
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))  # Hidden layer 2
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # Dropout layer 2
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))  # Hidden layer 3
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))  # Output layer

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model, "Deep_3Layer_128_256_128_ReLU_Adam"


def create_and_train_model(
        x_train, y_train, create_model_func, num_classes, fold, epochs=epochs_nr, batch_size=batch_size_nr):
    # Create the model using the provided function
    model, model_name = create_model_func((x_train.shape[1],), num_classes)

    # Train the model and save the history
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    # Convert the history.history dict to a pandas DataFrame
    hist_df = pd.DataFrame(history.history)

    # Save to csv
    hist_path = f'results/{model_name}/csv/history_fold_{fold}.csv'
    ensure_dir(hist_path)
    hist_df.to_csv(hist_path)

    return model, model_name

# def create_and_train_model(x_train, y_train, num_classes, fold):
#     # Define the model inside the function
#     model = Sequential1()
#     model.add(Input(shape=(x_train.shape[1],)))  # Input layer
#     model.add(Dense(32, activation='relu'))  # Hidden layer 1, 32 neurons, ReLU(Rectified Linear Unit) activation
#     model.add(Dense(64, activation='relu'))  # Hidden layer 2
#     model.add(Dense(num_classes, activation='softmax'))  # Output layer
#
#     # Compile the model
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#     # Train the model and save the history
#     history = model.fit(x_train, y_train, epochs=epochs_nr, batch_size=batch_size_nr)
#
#     # Convert the history.history dict to a pandas DataFrame
#     hist_df = pd.DataFrame(history.history)
#
#     # Save to csv
#     model_name = type(model).__name__
#     hist_df.to_csv(f'results/{model_name}/csv/history_fold_{fold}.csv')
#
#     return model, model_name
