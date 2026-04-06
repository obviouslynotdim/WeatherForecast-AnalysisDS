# 🌦️ Cambodia Weather Forecast Analysis

### Fundamentals of Data Science Project

**Cambodia Academy of Digital Technology (CADT)**
Bachelor of Computer Science

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-orange?style=for-the-badge&logo=pandas">
  <img src="https://img.shields.io/badge/Matplotlib-Visualization-green?style=for-the-badge">
  <img src="https://img.shields.io/badge/Open--Meteo-Weather%20API-purple?style=for-the-badge">
</p>

---

# 📌 Project Overview

This project analyzes **weather data in Cambodia** using the **Open-Meteo API** to understand climate patterns and support **weather forecasting analysis**.

Instead of static datasets, this project dynamically collects **historical weather data** using API requests and processes it with **Python, Pandas, and Matplotlib**.

The goal is to transform raw API data into **insightful climate trends and forecasting indicators**.

---

# 🎯 Objectives

* Collect historical weather data using Open-Meteo API
* Analyze key indicators such as:
  - Temperature (max/min)
  - Rainfall
  - Wind speed
* Identify **weather trends over time**
* Visualize climate patterns
* Support **basic forecasting insights**

---

# 🌎 Data Source

Data is collected from:

* **API:** Open-Meteo Weather API  
* **Website:** https://open-meteo.com/  
* **Type:** Historical weather data (archive API)

---

# 🧠 Methodology

## 1️⃣ Data Collection (API)

* Fetch data using `requests`
* Loop through multiple Cambodian provinces
* Collect daily weather data:
  - Temperature (max/min)
  - Rainfall
  - Wind speed

---

## 2️⃣ Data Preprocessing

* Combine data from all provinces
* Rename columns for clarity
* Convert date to datetime format
* Handle missing values if needed

---

## 3️⃣ Exploratory Data Analysis (EDA)

* Analyze weather trends over time
* Compare provinces
* Detect unusual patterns

---

## 4️⃣ Visualization

Using **Matplotlib**:

* Temperature trends
* Rainfall patterns
* Monthly averages
* Province comparisons

---

# 📊 Expected Outcomes

* Understand **climate patterns in Cambodia**
* Identify **seasonal trends**
* Provide **visual insights for forecasting**
* Demonstrate real-world **API-based data science workflow**

---

# 👨‍💻 Team Members

| Kuy Poly         | Chhorn Norakjed | Sophal Chanrat        |
| ---------------- | --------------- | --------------------- |
| **Te Chhenghab** | **Hak Kimly**   | **Sao Sethavathanak** |

---

# ⚙️ How to Run

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/cambodia-weather-forecast-analysis.git
cd cambodia-weather-forecast-analysis
````

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Jupyter Notebook

```bash
jupyter notebook
```

Open:

```
notebooks/weather_forecast_analysis.ipynb
```

You can also open the model-comparison notebooks:

```
notebooks/linear_regression/linear_regression.ipynb
notebooks/dicision_tree/dicision_tree.ipynb
notebooks/raindom_forest/random_forest.ipynb
```

---

# 📁 Folder Structure

```
cambodia-weather-forecast-analysis/
│
├── data/                           # Weather dataset CSV files
├── notebooks/                      # EDA and modeling notebooks
│   ├── weather_forecast_analysis.ipynb
│   ├── linear_regression/
│   ├── dicision_tree/
│   └── raindom_forest/
├── MODEL_ANALYSIS.md               # Model interpretation and findings
├── README.md
├── requirements.txt
├── requirement.txt                 # Legacy copy kept for compatibility
└── .gitignore
```

---

# 🌟 Features

* ✅ Real-time API-based data collection
* ✅ Multi-province Cambodia dataset
* ✅ Automated data pipeline
* ✅ EDA + visualization
* ✅ Regression model comparison and evaluation

---

# 🧾 Workflow Summary

1. Collect and consolidate weather data from Open-Meteo API.
2. Perform data cleaning and exploratory analysis in the main notebook.
3. Train and evaluate multiple regression baselines.
4. Compare results using RMSE, MAE, and R2.
5. Document interpretation, limitations, and improvement ideas.

---

<p align="center">
🌦️ From API Data to Climate Insights 🌍
</p>
