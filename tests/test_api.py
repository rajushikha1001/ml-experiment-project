import pytest
from src.api import app


class DummyModel:
    def predict(self, X):
        return [1]


@pytest.fixture
def client(monkeypatch):
    from src import api

    def fake_get_model():
        return DummyModel()

    monkeypatch.setattr(api, "get_model", fake_get_model)

    with app.test_client() as client:
        yield client


def test_predict(client):
    response = client.post(
        "/predict",
        json={"features": [1.0, 2.0, 3.0]}
    )

    assert response.status_code == 200
    assert response.get_json() == {"prediction": [1]}
