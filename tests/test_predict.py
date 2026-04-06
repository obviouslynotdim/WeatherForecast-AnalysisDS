import pytest

from src.models.predict import InferenceEngine


def test_inference_engine_requires_artifacts(tmp_path):
    missing_model = tmp_path / "model.joblib"
    missing_meta = tmp_path / "metadata.json"

    with pytest.raises(FileNotFoundError):
        InferenceEngine(model_path=missing_model, metadata_path=missing_meta)
