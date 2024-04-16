import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Instantiate the encoder
le = LabelEncoder()

df = pd.read_csv('datasets/predictive_maintenance.csv')

df['Failure Type'] = df['Failure Type'].map(
    {'No Failure': 0, 'Overstrain Failure': 1, 'Tool Wear Failure': 2, 'Power Failure': 3})

# Remove rows with NaN values
df = df.dropna()

X = df.drop('Failure Type', axis=1)
y = df['Failure Type']

# Split the data into training and test sets. Use 20% of the data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit and transform the second column of the DataFrame
X_train['Product ID'] = le.fit_transform(X_train['Product ID'])

# Get the labels that are in the training set
train_labels = set(le.classes_)


# Define a function to transform the labels
def transform_label(label):
    if label in train_labels:
        return le.transform([label])[0]
    else:
        return -1


# Transform the 'Product ID' column of the test set into integers
X_test['Product ID'] = X_test['Product ID'].apply(transform_label)

# Fit and transform the 'Type' column of the DataFrame
X_train['Type'] = le.fit_transform(X_train['Type'])
X_test['Type'] = le.transform(X_test['Type'])

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print('Decision Tree Accuracy:', accuracy_score(y_test, y_pred))

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print('Linear Regression Accuracy:', accuracy_score(y_test, y_pred.round()))

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print('Random Forest Accuracy:', accuracy_score(y_test, y_pred))
