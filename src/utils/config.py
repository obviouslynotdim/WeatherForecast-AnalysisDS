from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PREPROCESSORS_DIR = ARTIFACTS_DIR / "preprocessors"

DATASET_PATH = DATA_DIR / "cambodia_weather.csv"
MODEL_PATH = MODELS_DIR / "weather_model.joblib"
METADATA_PATH = PREPROCESSORS_DIR / "model_metadata.json"

MODEL_REGISTRY = {
	"linear_regression": MODELS_DIR / "weather_model_lr.joblib",
	"decision_tree": MODELS_DIR / "weather_model_dt.joblib",
	"random_forest": MODELS_DIR / "weather_model_rf.joblib",
}
COMPARE_METADATA_PATH = PREPROCESSORS_DIR / "model_comparison_metadata.json"
