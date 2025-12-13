import pytest
from src.model import MLModel
from sklearn.model_selection import train_test_split
from src.data_processing import load_data, process_data

def test_model_training():
    # Load and process data
    data = load_data('data/raw/dataset.csv')
    data = process_data(data)
    X_train, X_test, y_train, y_test = train_test_split(data.drop('target', axis=1), data['target'], test_size=0.2)
    
    # Train model
    model = MLModel()
    model.train(X_train, y_train)
    
    # Evaluate model
    accuracy = model.evaluate(X_test, y_test)
    
    assert accuracy > 0.7  # Assuming a good model should have > 70% accuracy
