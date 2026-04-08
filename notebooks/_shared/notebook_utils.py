from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def load_weather_csv(path: str | Path) -> pd.DataFrame:
    """Load weather CSV and normalize column names."""
    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]
    return df


def add_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add calendar + cyclical time features."""
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

    out["month_sin"] = np.sin(2.0 * np.pi * (out["month"].astype(float) - 1.0) / 12.0)
    out["month_cos"] = np.cos(2.0 * np.pi * (out["month"].astype(float) - 1.0) / 12.0)
    out["dayofweek_sin"] = np.sin(2.0 * np.pi * out["dayofweek"].astype(float) / 7.0)
    out["dayofweek_cos"] = np.cos(2.0 * np.pi * out["dayofweek"].astype(float) / 7.0)
    return out


def clean_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply lightweight quality filters used by training notebooks."""
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
        out = out[out["lat"].between(9.5, 14.5) & out["lon"].between(102.0, 108.0)]

    return out


def build_model_features(
    df: pd.DataFrame,
    target_col: str = "temp_max",
    province_col: str = "province",
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Build model-ready features with one-hot province encoding."""
    required = {province_col, target_col}
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
    base = work[keep_numeric + [province_col]].dropna().copy()

    provinces = sorted(base[province_col].astype(str).unique().tolist())
    one_hot = pd.get_dummies(base[province_col].astype(str), prefix=province_col)

    X = pd.concat([base[keep_numeric], one_hot], axis=1).astype(float)
    y = work.loc[X.index, target_col].astype(float)
    feature_columns = X.columns.tolist()
    return X, y, feature_columns, provinces


def random_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create random train/test split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )


def regression_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> dict:
    """Compute standardized regression metrics used in notebooks."""
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse": round(float(mean_squared_error(y_true, y_pred) ** 0.5), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 2),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
    }


def eda_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create compact EDA summary table."""
    if "date" in df.columns:
        date_min = str(pd.to_datetime(df["date"], errors="coerce").min().date())
        date_max = str(pd.to_datetime(df["date"], errors="coerce").max().date())
    else:
        date_min = "N/A"
        date_max = "N/A"

    province_value = "N/A"
    if "province" in df.columns:
        province_value = ", ".join(sorted(df["province"].astype(str).unique()))

    temp_avg_mean = "N/A"
    if {"temp_max", "temp_min"}.issubset(df.columns):
        temp_avg = (pd.to_numeric(df["temp_max"], errors="coerce") + pd.to_numeric(df["temp_min"], errors="coerce")) / 2.0
        temp_avg_mean = f"{temp_avg.mean():.2f}"

    return pd.DataFrame(
        [
            ["Rows", f"{len(df):,}"],
            ["Columns", str(df.shape[1])],
            ["Date start", date_min],
            ["Date end", date_max],
            ["Provinces", province_value],
            ["Missing values", str(int(df.isna().sum().sum()))],
            ["Duplicate rows", str(int(df.duplicated().sum()))],
            ["Mean temp_max (C)", f"{pd.to_numeric(df['temp_max'], errors='coerce').mean():.2f}" if "temp_max" in df.columns else "N/A"],
            ["Mean temp_min (C)", f"{pd.to_numeric(df['temp_min'], errors='coerce').mean():.2f}" if "temp_min" in df.columns else "N/A"],
            ["Mean temp_avg (C)", temp_avg_mean],
            ["Mean rain (mm)", f"{pd.to_numeric(df['rain'], errors='coerce').mean():.2f}" if "rain" in df.columns else "N/A"],
            ["Mean wind_speed", f"{pd.to_numeric(df['wind_speed'], errors='coerce').mean():.2f}" if "wind_speed" in df.columns else "N/A"],
        ],
        columns=["Metric", "Value"],
    )


def missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    """Create missing value report by column."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100 if len(df) else 0
    return pd.DataFrame(
        {
            "Column": df.columns,
            "Missing Count": missing.values,
            "Missing %": missing_pct.values if hasattr(missing_pct, "values") else np.zeros(len(df.columns)),
        }
    )
