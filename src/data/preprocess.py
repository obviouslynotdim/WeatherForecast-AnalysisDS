from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class DatasetBundle:
    features: pd.DataFrame
    target: pd.Series
    feature_columns: list[str]
    provinces: list[str]


def load_dataset(path: Path) -> pd.DataFrame:
    """Load CSV weather data and normalize column names."""
    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]
    return df


def add_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add calendar-based features extracted from the date column."""
    if date_col not in df.columns:
        raise ValueError(f"Missing required date column: {date_col}")

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    if out[date_col].isna().all():
        raise ValueError("Date parsing failed; all values are invalid.")

    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month
    out["day"] = out[date_col].dt.day
    out["dayofweek"] = out[date_col].dt.dayofweek
    # Cyclical encoding captures seasonality better than raw month/day indexes.
    out["month_sin"] = pd.Series(float("nan"), index=out.index, dtype="float64")
    out["month_cos"] = pd.Series(float("nan"), index=out.index, dtype="float64")
    out["dayofweek_sin"] = pd.Series(float("nan"), index=out.index, dtype="float64")
    out["dayofweek_cos"] = pd.Series(float("nan"), index=out.index, dtype="float64")

    valid_month = out["month"].notna()
    valid_dayofweek = out["dayofweek"].notna()
    month_angle = 2.0 * math.pi * (out.loc[valid_month, "month"].astype(float) - 1.0) / 12.0
    dayofweek_angle = 2.0 * math.pi * out.loc[valid_dayofweek, "dayofweek"].astype(float) / 7.0
    out.loc[valid_month, "month_sin"] = month_angle.map(math.sin)
    out.loc[valid_month, "month_cos"] = month_angle.map(math.cos)
    out.loc[valid_dayofweek, "dayofweek_sin"] = dayofweek_angle.map(math.sin)
    out.loc[valid_dayofweek, "dayofweek_cos"] = dayofweek_angle.map(math.cos)
    return out


def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply lightweight quality filters to improve training signal."""
    out = df.copy()

    numeric_cols = ["temp_max", "temp_min", "rain", "wind_speed", "lat", "lon"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "temp_max" in out.columns and "temp_min" in out.columns:
        out = out[out["temp_max"] >= out["temp_min"]]

    if "rain" in out.columns:
        out = out[out["rain"] >= 0]

    if "wind_speed" in out.columns:
        out = out[out["wind_speed"] >= 0]

    if "lat" in out.columns and "lon" in out.columns:
        # Keep records within Cambodia bounding box plus a small tolerance.
        out = out[out["lat"].between(9.5, 14.5) & out["lon"].between(102.0, 108.0)]

    return out


def build_features(
    df: pd.DataFrame,
    target_col: str = "temp_max",
    drop_cols: Sequence[str] = ("date",),
) -> DatasetBundle:
    """Create model-ready features with one-hot province encoding."""
    required = {"province", target_col}
    missing = required.difference(df.columns)
    if missing:
        missing_sorted = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_sorted}")

    work = df.dropna(subset=[target_col]).copy()
    numeric_cols = [
        "temp_min",
        "rain",
        "wind_speed",
        "lat",
        "lon",
        "year",
        "month",
        "day",
        "dayofweek",
        "month_sin",
        "month_cos",
        "dayofweek_sin",
        "dayofweek_cos",
    ]

    for col in numeric_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    keep_numeric = [c for c in numeric_cols if c in work.columns]
    base = work[keep_numeric + ["province"]].dropna().copy()

    provinces = sorted(base["province"].astype(str).unique().tolist())
    one_hot = pd.get_dummies(base["province"].astype(str), prefix="province")

    features = pd.concat([base[keep_numeric], one_hot], axis=1)
    features = features.astype(float)

    target = work.loc[features.index, target_col].astype(float)

    if drop_cols:
        _ = drop_cols  # reserved for future extension and explicit API clarity

    return DatasetBundle(
        features=features,
        target=target,
        feature_columns=features.columns.tolist(),
        provinces=provinces,
    )


def save_metadata(path: Path, target_col: str, feature_columns: list[str], provinces: list[str], train_metrics: dict) -> None:
    """Persist metadata required by inference and UI layer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "target_column": target_col,
        "feature_columns": feature_columns,
        "provinces": provinces,
        "metrics": train_metrics,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_metadata(path: Path) -> dict:
    """Load model metadata file used by prediction/UI components."""
    return json.loads(path.read_text(encoding="utf-8"))
