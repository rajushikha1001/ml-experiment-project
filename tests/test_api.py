import pytest
from flask import json
from src.api import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_predict(client):
    response = client.post('/predict', json={'features': [5.1, 3.5, 1.4, 0.2]})
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data
