#We define our model and training functions here

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class MLModel:
    def __init__(self):
        """Initialize the RandomForestClassifier with given parameters."""
        self.model = RandomForestClassifier(n_estimators=100)   

    def train(self, X_train, y_train):
        """Train the model on the training data."""
        self.model.fit(X_train, y_train)    

    def evaluate(self, X_test, y_test):
        """Evaluate the model's performance on the test data."""
        prey_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, prey_pred)
        return accuracy