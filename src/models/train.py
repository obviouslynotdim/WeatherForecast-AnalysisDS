from __future__ import annotations

import argparse
import json
from os import cpu_count

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.data.preprocess import add_time_features, build_features, clean_weather_data, load_dataset, save_metadata
from src.utils.config import DATASET_PATH, METADATA_PATH, MODEL_PATH


def train_model(
    target_col: str = "temp_max",
    test_size: float = 0.2,
    random_state: int = 42,
    n_jobs: int | None = None,
) -> dict:
    """Train baseline model and persist artifacts for inference."""
    df = load_dataset(DATASET_PATH)
    df = add_time_features(df)
    df = clean_weather_data(df)
    bundle = build_features(df, target_col=target_col)

    effective_n_jobs = n_jobs
    if effective_n_jobs is None:
        # Cap default parallelism to keep laptops responsive during training.
        effective_n_jobs = max(1, min(2, cpu_count() or 1))

    X_train, X_test, y_train, y_test = train_test_split(
        bundle.features,
        bundle.target,
        test_size=test_size,
        random_state=random_state,
    )

    model = RandomForestRegressor(
        n_estimators=220,
        max_depth=24,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=effective_n_jobs,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(mean_squared_error(y_test, preds) ** 0.5),
        "r2": float(r2_score(y_test, preds)),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "n_jobs": int(effective_n_jobs),
        "n_estimators": int(model.n_estimators),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    save_metadata(
        path=METADATA_PATH,
        target_col=target_col,
        feature_columns=bundle.feature_columns,
        provinces=bundle.provinces,
        train_metrics=metrics,
    )

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train weather prediction model.")
    parser.add_argument("--target", default="temp_max", help="Target column to predict (default: temp_max)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data used for test split")
    parser.add_argument("--seed", type=int, default=42, help="Random state")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="CPU workers for training. Default keeps laptop responsive.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="Deprecated compatibility flag. Ignored to avoid heavy training.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=0,
        help="Deprecated compatibility flag. Ignored to avoid heavy training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cv_folds or args.n_iter:
        print(
            "[info] --cv-folds and --n-iter are ignored in this lightweight trainer to prevent laptop lag/crashes."
        )
    metrics = train_model(
        target_col=args.target,
        test_size=args.test_size,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
