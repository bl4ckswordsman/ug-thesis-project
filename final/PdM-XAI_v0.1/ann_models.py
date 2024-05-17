import time

from keras import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
import pandas as pd
from tensorflow.keras.regularizers import l2

from utils import ensure_dir

epochs_nr = 50
batch_size_nr = 10


#     Artificial Neural Networks (ANNs),
#      specifically Feedforward Neural Networks (FNNs)
#      or Multilayer Perceptrons (MLPs):

def create_model1(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))  # Input layer
    model.add(Dense(32, activation='relu'))  # Hidden layer 1, 32 neurons, ReLU(Rectified Linear Unit) activation
    model.add(Dense(64, activation='relu'))  # Hidden layer 2
    model.add(Dense(num_classes, activation='softmax'))  # Output layer

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model, "Seq_2Layer_32_64_ReLU_Adam"


def create_model2(input_shape, num_classes):
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


def create_model3(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))  # Input layer
    model.add(
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)))  # Hidden layer 1, 128 neurons, ReLU activation
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


def create_model4(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))  # Input layer
    model.add(Dense(32, activation='relu'))  # Hidden layer
    model.add(Dense(num_classes, activation='softmax'))  # Output layer

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model, "Simple_1Layer_32_ReLU_Adam"


def create_model5(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=input_shape))  # Input layer
    model.add(Dense(128, activation='relu'))  # Hidden layer 1
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # Dropout layer 1
    model.add(Dense(256, activation='relu'))  # Hidden layer 2
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # Dropout layer 2
    model.add(Dense(256, activation='relu'))  # Hidden layer 3
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # Dropout layer 3
    model.add(Dense(128, activation='relu'))  # Hidden layer 4
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))  # Output layer

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model, "Complex_4Layer_128_256_256_128_ReLU_Adam"


def create_and_train_model(
        x_train, y_train, create_model_func, num_classes, fold, epochs=epochs_nr, batch_size=batch_size_nr):
    # Create the model using the provided function
    model, model_name = create_model_func((x_train.shape[1],), num_classes)

    # Record the start time
    start_time = time.time()

    # Train the model and save the history
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

    # Record the end time
    end_time = time.time()

    # Calculate the training time
    training_time = end_time - start_time

    # Convert the history.history dict to a pandas DataFrame
    hist_df = pd.DataFrame(history.history)

    # Add the training time to the DataFrame
    hist_df['training_time'] = training_time

    # Save to csv
    hist_path = f'results/{model_name}/csv/history_fold_{fold}.csv'
    ensure_dir(hist_path)
    hist_df.to_csv(hist_path)

    return model, model_name

