from flask import Flask, request, jsonify
import mlflow
import mlflow.sklearn

app = Flask(__name__)

# Load the trained model
model = mlflow.sklearn.load_model('models/model')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get input data
    features = data['features']
    prediction = model.predict([features])  # Predict using the loaded model
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
