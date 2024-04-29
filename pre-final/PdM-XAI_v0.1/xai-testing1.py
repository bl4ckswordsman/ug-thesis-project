from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('datasets/predictive_maintenance.csv')

# Convert the 'Type' column into dummy/indicator variables
df = pd.get_dummies(df, columns=['Type'])

# Create a label encoder
le = LabelEncoder()

# Fit the encoder to the 'Failure Type' column
le.fit(df['Failure Type'])

# Transform the 'Failure Type' column into integer labels
y = le.transform(df['Failure Type'].values)

num_classes = len(le.classes_)

# Define the feature matrix
X = df.drop(['Failure Type', 'Product ID'], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to categorical one-hot encoding
y_train_categorical = to_categorical(y_train, num_classes=num_classes)
y_test_categorical = to_categorical(y_test, num_classes=num_classes)

# Define the model
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))  # Input layer
model.add(Dense(32, activation='relu'))  # Hidden layer 1
model.add(Dense(64, activation='relu'))  # Hidden layer 2
model.add(Dense(num_classes, activation='softmax'))  # Output layer

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train_categorical, epochs=50, batch_size=10)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test_categorical)
print('ANN Accuracy:', accuracy)