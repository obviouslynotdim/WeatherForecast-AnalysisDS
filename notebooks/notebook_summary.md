# Notebook Work Summary

## What Was Implemented
- Shared reusable notebook logic in notebooks/_shared/notebook_utils.py
- Unified training flow across:
	- notebooks/random_forest/random_forest.ipynb
	- notebooks/linear_regression/linear_regression.ipynb
	- notebooks/decision_tree/dicision_tree.ipynb
- Filled model analysis notebooks with executed metrics and feature insights
- Kept model comparison notebook focused on comparison table + charts
- Added key EDA visualizations

## Common Training Pipeline
1. Load and normalize weather data
2. Add calendar and cyclical time features
3. Apply data quality filters
4. Build features with province one-hot encoding
5. Split data randomly (test_size=0.2, random_state=42)
6. Train model and compute metrics

## Model Results
Source: artifacts/preprocessors/model_comparison_metadata.json

| Model | RMSE | MAE | R2 | Train Rows | Test Rows |
|---|---:|---:|---:|---:|---:|
| Random Forest | 1.1331 | 0.8590 | 0.8624 | 16,456 | 4,114 |
| Decision Tree | 1.4806 | 1.1036 | 0.7650 | 16,456 | 4,114 |
| Linear Regression | 1.5369 | 1.1926 | 0.7468 | 16,456 | 4,114 |

Best model by RMSE: Random Forest

- EDA:
	- Correlation heatmap
	- Monthly temperature seasonality boxplot
- Model comparison:
	- RMSE, MAE, and R2 bar charts

## Final State
- Reusable notebook utilities in one place
- Consistent training/feature pipeline across 3 model notebooks
- Analysis notebooks populated with real executed outputs
- Clear visual support for both EDA and model comparison
