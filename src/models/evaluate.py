from __future__ import annotations

from src.models.train import train_model


if __name__ == "__main__":
    metrics = train_model()
    print("Model retrained. Metrics:")
    for key, value in metrics.items():
        print(f"- {key}: {value}")
