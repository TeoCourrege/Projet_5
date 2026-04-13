import pytest
import numpy as np
import joblib
from src.model.train import train_pipeline


def test_pipeline_training(tmp_path, sample_data):
    """Pipeline trains without error and produces a valid model file."""
    csv_path = tmp_path / "df_final.csv"
    model_path = tmp_path / "model.pkl"

    sample_data.to_csv(csv_path, index=False)
    pipeline = train_pipeline(csv_path=str(csv_path), model_path=str(model_path))

    assert model_path.exists(), "model.pkl should be created"

    X = sample_data.drop(columns=["a_quitte_l_entreprise", "id"])
    preds = pipeline.predict(X)

    assert len(preds) == len(X)
    assert preds[0] in [0, 1]


def test_pipeline_output_shape(tmp_path, sample_data):
    """Preprocessor produces a numpy array with the right number of rows."""
    csv_path = tmp_path / "df.csv"
    model_path = tmp_path / "model.pkl"

    sample_data.to_csv(csv_path, index=False)
    pipeline = train_pipeline(csv_path=str(csv_path), model_path=str(model_path))

    X = sample_data.drop(columns=["a_quitte_l_entreprise", "id"])
    transformed = pipeline.named_steps["preprocessing"].transform(X)

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == X.shape[0]


def test_model_persistence(tmp_path, sample_data):
    """Model can be saved and reloaded, producing identical predictions."""
    csv_path = tmp_path / "df.csv"
    model_path = tmp_path / "model.pkl"

    sample_data.to_csv(csv_path, index=False)
    pipeline = train_pipeline(csv_path=str(csv_path), model_path=str(model_path))

    loaded = joblib.load(str(model_path))
    X = sample_data.drop(columns=["a_quitte_l_entreprise", "id"])

    assert list(pipeline.predict(X)) == list(loaded.predict(X))


def test_predict_proba_shape(tmp_path, sample_data):
    """predict_proba returns probabilities for 2 classes."""
    csv_path = tmp_path / "df.csv"
    model_path = tmp_path / "model.pkl"

    sample_data.to_csv(csv_path, index=False)
    pipeline = train_pipeline(csv_path=str(csv_path), model_path=str(model_path))

    X = sample_data.drop(columns=["a_quitte_l_entreprise", "id"])
    proba = pipeline.predict_proba(X)

    # on ne predit que la proba d'appartenir a la classe (a_quitte_lentreprise == true)
    assert proba.shape == (len(X), 2) 
    assert 0.0 <= proba[0][0] <= 1.0
    assert 0.0 <= proba[0][1] <= 1.0
