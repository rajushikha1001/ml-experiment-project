import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """Load data from a CSV file."""
    data = pd.read_csv(filepath)
    return data 

def process_data(data):
    """Process the loaded data."""
    # Example: drop missing values
    data = data.fillna(data.mean())
    data = pd.get_dummies(data, drop_first=True)   
    return data

def split_data(data, test_size=0.2):
    """Split data into training and testing sets."""
    X = data.drop('target', axis=1)
    y = data['target']
    return train_test_split(X, y, test_size=test_size)