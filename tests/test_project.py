"""Small integration checks against the artifact produced by python train.py."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from predict import app
from train import ROOT, FEATURES, SEED, load_data


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as active:
        yield active


def test_data_and_split():
    data = load_data()
    assert data.shape == (768, 9)
    assert list(data.columns) == FEATURES + ["heating_load"]
    assert np.isfinite(data.to_numpy()).all()
    training, test = train_test_split(data, test_size=0.2, random_state=SEED)
    assert (len(training), len(test)) == (614, 154)
    assert set(training.index).isdisjoint(test.index)
    assert len(set(training.index) | set(test.index)) == 768


def test_saved_model_reproduces_metrics():
    data = load_data()
    _, test = train_test_split(data, test_size=0.2, random_state=SEED)
    model = joblib.load(ROOT / "models/heating_model.joblib")
    predictions = model.predict(test[FEATURES])
    assert predictions.shape == (154,)
    assert np.isfinite(predictions).all()
    expected = json.loads((ROOT / "results.json").read_text())["test_metrics"]
    assert root_mean_squared_error(test.heating_load, predictions) == pytest.approx(expected["rmse"])
    assert mean_absolute_error(test.heating_load, predictions) == pytest.approx(expected["mae"])
    assert r2_score(test.heating_load, predictions) == pytest.approx(expected["r2"])


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction_matches_pipeline(client):
    sample = json.loads((ROOT / "examples/building.json").read_text())
    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    assert set(response.json()) == {"heating_load"}
    model = joblib.load(ROOT / "models/heating_model.joblib")
    expected = float(model.predict(pd.DataFrame([sample]))[0])
    assert response.json()["heating_load"] == pytest.approx(expected)


def test_invalid_input(client):
    sample = json.loads((ROOT / "examples/building.json").read_text())
    for payload in [{}, {**sample, "orientation": 99}]:
        assert client.post("/predict", json=payload).status_code == 422
