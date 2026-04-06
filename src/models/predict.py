from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.data.preprocess import load_metadata
from src.utils.config import COMPARE_METADATA_PATH, METADATA_PATH, MODEL_PATH


class InferenceEngine:
    """Load trained artifacts and run model inference."""

    def __init__(self, model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata artifact not found: {metadata_path}")

        self.model = joblib.load(model_path)
        self.metadata = load_metadata(metadata_path)
        self.feature_columns = self.metadata["feature_columns"]

    def build_input_frame(
        self,
        temp_min: float,
        rain: float,
        wind_speed: float,
        lat: float,
        lon: float,
        year: int,
        month: int,
        day: int,
        dayofweek: int,
        province: str,
    ) -> pd.DataFrame:
        row = {
            "temp_min": float(temp_min),
            "rain": float(rain),
            "wind_speed": float(wind_speed),
            "lat": float(lat),
            "lon": float(lon),
            "year": int(year),
            "month": int(month),
            "day": int(day),
            "dayofweek": int(dayofweek),
            "month_sin": math.sin(2.0 * math.pi * (int(month) - 1) / 12.0),
            "month_cos": math.cos(2.0 * math.pi * (int(month) - 1) / 12.0),
            "dayofweek_sin": math.sin(2.0 * math.pi * int(dayofweek) / 7.0),
            "dayofweek_cos": math.cos(2.0 * math.pi * int(dayofweek) / 7.0),
        }

        for col in self.feature_columns:
            if col.startswith("province_"):
                row[col] = 1.0 if col == f"province_{province}" else 0.0

        frame = pd.DataFrame([row])
        frame = frame.reindex(columns=self.feature_columns, fill_value=0.0)
        return frame.astype(float)

    def predict(self, **kwargs) -> float:
        features = self.build_input_frame(**kwargs)
        pred = self.model.predict(features)[0]
        return float(pred)


class MultiModelInferenceEngine:
    """Load multiple trained artifacts and run side-by-side inference."""

    def __init__(self, metadata_path: Path = COMPARE_METADATA_PATH):
        if not metadata_path.exists():
            raise FileNotFoundError(
                "Comparison metadata not found. Train comparison artifacts with: "
                "python -m src.models.train_compare"
            )

        self.metadata = load_metadata(metadata_path)
        self.feature_columns = self.metadata["feature_columns"]

        models_meta: dict[str, Any] = self.metadata.get("models", {})
        if not models_meta:
            raise ValueError("Comparison metadata has no model entries.")

        self.models: dict[str, Any] = {}
        for model_key, details in models_meta.items():
            model_path = Path(details["model_path"])
            if not model_path.exists():
                raise FileNotFoundError(f"Model artifact not found for {model_key}: {model_path}")
            self.models[model_key] = joblib.load(model_path)

    def build_input_frame(
        self,
        temp_min: float,
        rain: float,
        wind_speed: float,
        lat: float,
        lon: float,
        year: int,
        month: int,
        day: int,
        dayofweek: int,
        province: str,
    ) -> pd.DataFrame:
        row = {
            "temp_min": float(temp_min),
            "rain": float(rain),
            "wind_speed": float(wind_speed),
            "lat": float(lat),
            "lon": float(lon),
            "year": int(year),
            "month": int(month),
            "day": int(day),
            "dayofweek": int(dayofweek),
            "month_sin": math.sin(2.0 * math.pi * (int(month) - 1) / 12.0),
            "month_cos": math.cos(2.0 * math.pi * (int(month) - 1) / 12.0),
            "dayofweek_sin": math.sin(2.0 * math.pi * int(dayofweek) / 7.0),
            "dayofweek_cos": math.cos(2.0 * math.pi * int(dayofweek) / 7.0),
        }

        for col in self.feature_columns:
            if col.startswith("province_"):
                row[col] = 1.0 if col == f"province_{province}" else 0.0

        frame = pd.DataFrame([row])
        frame = frame.reindex(columns=self.feature_columns, fill_value=0.0)
        return frame.astype(float)

    def predict_all(self, **kwargs) -> dict[str, float]:
        features = self.build_input_frame(**kwargs)
        results: dict[str, float] = {}
        for model_key, model in self.models.items():
            pred = model.predict(features)[0]
            results[model_key] = float(pred)
        return results
