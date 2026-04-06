from __future__ import annotations

import argparse
import json
from os import cpu_count

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from src.data.preprocess import add_time_features, build_features, clean_weather_data, load_dataset, save_metadata
from src.utils.config import COMPARE_METADATA_PATH, DATASET_PATH, MODEL_REGISTRY


def _evaluate(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_and_compare_models(
    target_col: str = "temp_max",
    test_size: float = 0.2,
    random_state: int = 42,
    n_jobs: int | None = None,
) -> dict:
    """Train three regression models and persist artifacts + comparison metadata."""
    df = load_dataset(DATASET_PATH)
    df = add_time_features(df)
    df = clean_weather_data(df)
    bundle = build_features(df, target_col=target_col)

    effective_n_jobs = n_jobs
    if effective_n_jobs is None:
        effective_n_jobs = max(1, min(2, cpu_count() or 1))

    X_train, X_test, y_train, y_test = train_test_split(
        bundle.features,
        bundle.target,
        test_size=test_size,
        random_state=random_state,
    )

    model_specs = {
        "linear_regression": LinearRegression(),
        "decision_tree": DecisionTreeRegressor(max_depth=15, random_state=random_state),
        "random_forest": RandomForestRegressor(
            n_estimators=220,
            max_depth=24,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=effective_n_jobs,
        ),
    }

    models_section: dict[str, dict] = {}
    for model_key, model in model_specs.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = _evaluate(y_test, preds)
        metrics["train_rows"] = int(X_train.shape[0])
        metrics["test_rows"] = int(X_test.shape[0])
        metrics["split_strategy"] = "random"

        model_path = MODEL_REGISTRY[model_key]
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

        models_section[model_key] = {
            "model_type": model.__class__.__name__,
            "model_path": str(model_path),
            "metrics": metrics,
        }

    best_model = min(models_section, key=lambda k: models_section[k]["metrics"]["rmse"])

    comparison_payload = {
        "target_column": target_col,
        "feature_columns": bundle.feature_columns,
        "provinces": bundle.provinces,
        "best_model": best_model,
        "models": models_section,
    }

    save_metadata(
        path=COMPARE_METADATA_PATH,
        target_col=target_col,
        feature_columns=bundle.feature_columns,
        provinces=bundle.provinces,
        train_metrics={"comparison_ready": True, "best_model": best_model},
    )
    # Overwrite with comparison-specific schema for multi-model inference.
    COMPARE_METADATA_PATH.write_text(json.dumps(comparison_payload, indent=2), encoding="utf-8")

    return comparison_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare three weather prediction models.")
    parser.add_argument("--target", default="temp_max", help="Target column to predict (default: temp_max)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data used for test split")
    parser.add_argument("--seed", type=int, default=42, help="Random state")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="CPU workers for random forest training. Default keeps laptop responsive.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_and_compare_models(
        target_col=args.target,
        test_size=args.test_size,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
