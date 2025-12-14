import mlflow
import mlflow.sklearn
from src.model import MLModel
from src.data_processing import load_data, process_data, split_data

def track_experiment():
    """Track the machine learning experiment using MLflow."""
    # Load and process data
    data = load_data('data/raw/dataset.csv')
    data = process_data(data)
    X_train, X_test, y_train, y_test = split_data(data)

    mlflow.start_run()
    # Initialize model
    model = MLModel()
    model.train(X_train, y_train)
    
    # Evaluate the model
    accuracy = model.evaluate(X_test, y_test)

    # Log parameters, metrics, and the model itself
    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model.model, "model")
 
    # End experiment
    mlflow.end_run()
    return accuracy