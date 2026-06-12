# Nifty50 Intelligent Investment Platform

An end to end, institutional grade AI platform designed to transform raw Nifty50 market data into actionable investment intelligence. This system combines robust tree based Machine Learning (XGBoost), Modern Portfolio Theory, and advanced Model Explainability (SHAP) into a single, cohesive Streamlit dashboard.

---

## System Architecture

Our closed loop AI system operates in four distinct stages:
1. **Data and Feature Engineering:** Cleans historical Nifty50 data and engineers 10+ technical indicators (RSI, MACD, Bollinger Bands, Volatility, Z-Scores) while preventing look-ahead bias.
2. **Predictive Modeling Engine:** Utilizes an Extreme Gradient Boosting (XGBoost) architecture validated via strict Time Series cross validation to predict the probability of positive 5 day forward returns. 
3. **Automated Portfolio Construction:** Integrates directly with the AI engine to isolate the Top 10 High Conviction Buy Candidates and optimizes their capital allocation using PyPortfolioOpt (Sharpe Ratio maximization).
4. **AI Transparency:** Deploys Game Theoretic Shapley values (SHAP TreeExplainer) to break open the "black box" and mathematically prove *why* the model makes its decisions.

---

## Directory Structure

```text
nifty-intelligence-platform/
│
├── data/
│   ├── raw.zip              # Original historical CSVs before running any file first unzip the file in the folder itself
│   ├── processed/           # Cleaned data with engineered features
│   └── xgb_predictions.csv  # Live output from the XGBoost predictor engine
│
├── models/                  # Stored .joblib XGBoost models
│
├── src/
│   ├── data_pipeline.py     # Making the processes data
│   ├── features.py          # Data ingestion and feature engineering
│   ├── predictor.py         # XGBoost training and prediction engine
│   ├── explainability.py    # SHAP TreeExplainer routing logic
│   └── portfolio.py         # PyPortfolioOpt optimization logic
│
├── app/
│   └── main.py              # Interactive Streamlit Web Application
│
└── README.md                # Project documentation

```

---

## Installation and Environment Setup

Due to strict binary dependencies between numerical libraries, please follow these installation instructions exactly to ensure a stable environment.

**1. Create a virtual environment (Python 3.10 or 3.11 recommended):**

```bash
conda create -n nifty_env python=3.11 -y
conda activate nifty_env
```

**2. Install Core Dependencies:**
*Note: We strictly pin Numpy < 2.0 and XGBoost 2.0.3 to ensure full compatibility with the SHAP explainability library.*

```bash
pip install "numpy<2" pandas "xgboost==2.0.3" scikit-learn
pip install scipy cvxpy PyPortfolioOpt shap matplotlib streamlit
```

**3. Install PyArrow (macOS specific, if applicable):**

```bash
pip install --force-reinstall --no-cache-dir pyarrow
```

---

## Execution Guide

To run the platform from scratch, execute the following scripts sequentially from your terminal at the root directory:

**Step 1: Engineer the Features**
Cleans the raw data and builds the technical indicators.

```bash
python src/features.py
```

**Step 2: Train the XGBoost Engine**
Trains 50 unique tree based models, validates them using Time-Series splits, saves the `.joblib` files, and generates the live market forecast.

```bash
python src/predictor.py
```

**Step 3: Launch the Dashboard**
Start the interactive Streamlit application.

```bash
streamlit run app/main.py
```

---

## Dashboard Features

* **Tab 1: Market Forecast:** A live heatmap ranking the Nifty50 universe by their mathematical probability of generating positive returns over the next 5 days.
* **Tab 2: Portfolio Builder:** A Closed Loop allocation system. It automatically pulls the AI's Top 10 Buy signals and balances the capital based on your selected risk profile (Conservative, Balanced, Aggressive).
* **Tab 3: AI Explainability:** Breaks open the "black box." Select any stock to see a dynamic SHAP Matplotlib chart showing exactly which technical indicators drove the AI's latest prediction.
* **Tab 4: Risk and Anomalies:** Scans the latest day of trading data to flag extreme volatility and unusual trading volume using statistical Z Scores.

---
