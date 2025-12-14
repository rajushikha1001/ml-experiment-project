from flask import Flask, request, jsonify
import mlflow.sklearn

app = Flask(__name__)

_model = None  # private cache


def get_model():
    """
    Lazily load ML model.
    This prevents crashes during import, testing, and CI.
    """
    global _model
    if _model is None:
        _model = mlflow.sklearn.load_model("models/model")
    return _model


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    features = payload["features"]

    model = get_model()
    prediction = model.predict([features])

    return jsonify({"prediction": prediction.tolist()})
