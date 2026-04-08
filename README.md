# 🌦️ Cambodia Weather Forecast Analysis

### Fundamentals of Data Science Project

**Cambodia Academy of Digital Technology (CADT)**  
Bachelor of Computer Science

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-orange?style=for-the-badge&logo=pandas">
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-yellow?style=for-the-badge&logo=scikitlearn">
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Open--Meteo-Weather%20API-purple?style=for-the-badge">
</p>

---

## 📌 Project Overview

This project analyzes Cambodia weather patterns from Open-Meteo data and builds a three-model regression comparison pipeline for next-day maximum temperature prediction.

The workflow includes:
- API-based historical weather collection
- Data preparation and quality filtering
- Exploratory Data Analysis (EDA)
- Training and evaluation of three models:
  - Linear Regression
  - Decision Tree
  - Random Forest
- Side-by-side model comparison using RMSE, MAE, and R2

---

## 🎯 Objectives

- Collect historical weather data for multiple Cambodian provinces
- Analyze temperature, rainfall, and wind behavior over time
- Engineer reusable forecasting features (time + cyclical + geospatial + province)
- Train multiple baseline models with a consistent pipeline
- Compare model performance and expose results in notebooks and Gradio UI

---

## 🌎 Data Source

- API: Open-Meteo Weather API
- Website: https://open-meteo.com/
- Type: Historical weather data (archive API)

---

## 🧠 Methodology

### 1) Data Collection
- Fetch daily archive data via API requests
- Loop through selected provinces
- Store standardized fields: date, temp_max, temp_min, rain, wind_speed, province, lat, lon

### 2) Feature Engineering and Cleaning
- Normalize columns and parse date
- Add time features: year, month, day, dayofweek
- Add cyclical features: month_sin, month_cos, dayofweek_sin, dayofweek_cos
- Apply quality filters (temperature consistency, non-negative rain/wind, Cambodia geo bounds)
- One-hot encode province

### 3) Modeling
- Train/test split: random 80/20, seed 42
- Models:
  - LinearRegression
  - DecisionTreeRegressor (max_depth=15)
  - RandomForestRegressor (n_estimators=220, max_depth=24, min_samples_leaf=2)

### 4) Evaluation
- Metrics: RMSE, MAE, R2
- Compare all models from saved metadata in one table and chart set

---

## 📓 Notebooks

Main notebooks:
- notebooks/eda.ipynb
- notebooks/weather_forecast_analysis.ipynb
- notebooks/model_comparisons.ipynb

Model training notebooks:
- notebooks/linear_regression/linear_regression.ipynb
- notebooks/decision_tree/dicision_tree.ipynb
- notebooks/random_forest/random_forest.ipynb

Model analysis notebooks:
- notebooks/linear_regression/linear_regression_analysis.ipynb
- notebooks/decision_tree/decision_tree_analysis.ipynb
- notebooks/random_forest/random_forest_analysis.ipynb

Notebook summary:
- NOTEBOOK_ANALYSIS.md

---

## ⚙️ How to Run

### 1) Clone repository

```bash
git clone https://github.com/your-username/cambodia-weather-forecast-analysis.git
cd cambodia-weather-forecast-analysis
```

### 2) Create and activate virtual environment (recommended)

```bash
python -m venv .venv
```

PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Train all three models and generate comparison artifacts

```bash
python -m src.models.train_compare --target temp_max
```

Generated artifacts:

```text
artifacts/models/weather_model_lr.joblib
artifacts/models/weather_model_dt.joblib
artifacts/models/weather_model_rf.joblib
artifacts/preprocessors/model_comparison_metadata.json
```

Optional single-model trainer:

```bash
python -m src.models.train --target temp_max
```

### 5) Launch notebooks

```bash
jupyter notebook
```

### 6) Launch Gradio app

```bash
python app/gradio_app.py
```

---

## 📊 Visualization Highlights

EDA notebook includes:
- Province temperature distribution
- Monthly rainfall trend
- Yearly temperature trend by province
- Correlation heatmap of weather/time features
- Monthly temperature seasonality boxplot

Model comparison notebook includes:
- Comparison table (RMSE, MAE, R2, row counts, best model flag)
- RMSE bar chart
- MAE bar chart
- R2 bar chart

---

## 📁 Folder Structure

```text
WeatherForecast-AnalysisDS/
├── app/
│   └── gradio_app.py
├── artifacts/
│   ├── models/
│   └── preprocessors/
├── data/
│   ├── raw/
│   └── cambodia_weather.csv
├── notebooks/
│   ├── _shared/
│   │   └── notebook_utils.py
│   ├── decision_tree/
│   ├── linear_regression/
│   ├── random_forest/
│   ├── eda.ipynb
│   ├── model_comparisons.ipynb
│   └── weather_forecast_analysis.ipynb
├── src/
│   ├── data/
│   ├── models/
│   └── utils/
├── tests/
├── MODEL_ANALYSIS.md
├── NOTEBOOK_ANALYSIS.md
├── README.md
└── requirements.txt
```

---

## 👨‍💻 Team Members

| Kuy Poly         | Chhorn Norakjed | Sophal Chanrat        |
| ---------------- | --------------- | --------------------- |
| **Te Chhenghab** | **Hak Kimly**   | **Sao Sethavathanak** |

---

<p align="center">
🌦️ From API Data to Climate Insights 🌍
</p>
