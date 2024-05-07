import pandas as pd
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess_data():
    # Load the data
    df = pd.read_csv('datasets/predictive_maintenance.csv')

    # Preprocess the data
    df = pd.get_dummies(df, columns=['Type'])
    le = LabelEncoder()
    df['Failure Type'] = le.fit_transform(df['Failure Type'])
    X = df.drop(['Failure Type', 'Product ID'], axis=1)
    y = df['Failure Type']

    return X, y, le
