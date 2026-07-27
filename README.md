# Influenza and COVID case prediction using machine learning

This [dashboard](https://tommbk50-hub.github.io/UK-respiratory-illness-case-predictions-using-machine-learning/dashboard.html) uses a Hybrid Machine Learning strategy (HistGradientBoostingRegressor) trained on historical surveillance data from the [UK Health Security Agency website](https://ukhsa-dashboard.data.gov.uk/) to provide a comprehensive 52-week forecast for Influenza (Flu) and COVID-19 in England using machine learning to predict future patterns. By analysing trends in PCR Positivity and Hospital Admissions, these models aim to support healthcare resource planning—helping hospitals and A&E departments anticipate potential winter surges.

The system is fully automated, retrieving live data from the UKHSA public API each week, then iteratively retraining the model every week via GitHub Actions, ensuring the forecast is always based on the latest available statistics.

# UKHSA Respiratory Disease Forecaster

## Overview
This Python script is designed to automate the extraction, analysis, and forecasting of respiratory disease data (specifically COVID-19 and Influenza) in England. It pulls historical data from a government health API, uses machine learning to predict future trends, and packages the results into a JSON file, intended to serve as the backend data source for a dashboard.

---

## How It Works

### 1. Configuration and Setup
* **Metrics Definition:** The script defines a `METRICS` dictionary that configures five specific health tracking targets: Influenza PCR positivity, Influenza hospital admission rates, Influenza ICU/HDU admission rates, COVID-19 PCR positivity, and COVID-19 weekly hospital admissions.
* **API Targeting:** It constructs an `API_TEMPLATE` URL to pull this data directly from the UK Health Security Agency (UKHSA) dashboard API.

### 2. Data Fetching
* **Robust Web Requests:** The script uses the `requests` library wrapped with a `Retry` strategy to handle potential network errors (such as HTTP 429 Too Many Requests or 500 Server Errors) when calling the API.
* **Pagination Handling:** It loops through paginated API results (`page_size=365`) until all available historical data for a given metric is downloaded.
* **Data Cleaning:** Once fetched, the data is loaded into a `pandas` DataFrame, converted to datetime objects, and aggregated into a weekly format (ending on Sundays) by either summing or averaging the values, depending on the metric's configuration.

### 3. Machine Learning Forecast
To predict future health metrics up to 52 weeks in advance, the script uses a two-step machine learning approach via `HistGradientBoostingRegressor` from the `sklearn` library:
* **Seasonal Baseline:** It first trains a model on the week number and month to capture predictable seasonal trends (e.g., winter flu spikes).
* **Residual (Lag) Modeling:** It calculates the "residual" (the difference between the actual data and the seasonal prediction) and trains a second model to predict these residuals based on the data from 1, 2, and 3 weeks prior.
* **Confidence Intervals:** It trains two additional quantile regression models (at the 5% and 95% quantiles) to generate upper and lower bounds for its predictions, providing a measured margin of error.
* **Feature Importance:** It calculates `permutation_importance` to determine which lagged weeks were most influential in generating the predictions.

### 4. Accuracy Evaluation

The `evaluate_accuracy` function in this script implements a rigorous **walk-forward validation** (or time-series backtesting) approach rather than a simple train/test split. This ensures the performance metrics accurately reflect how the model would have performed in a live production environment without introducing data leakage. 

Here is an expanded breakdown of the mechanics inside this module:

* **Walk-Forward Cross-Validation:** The script defaults to evaluating the last 52 weeks of data. It iterates backward through time step-by-step. For each target week in that 52-week window, it establishes a strict cutoff point (`train_end_index`).
* **Strict Temporal Isolation:** To prevent data leakage, it creates a fresh training slice containing *only* the data that existed prior to the target week. The model has no access to future trends during its evaluation phase.
* **Iterative Model Retraining:** For every single week in the evaluation loop, the script completely refits both `HistGradientBoostingRegressor` models from scratch. It retrains the seasonal base model on the historical weeks and months, calculates the historical residuals, and then retrains the lag-based residual model. 
* **Prediction and Assembly:** The script simulates a live prediction for the target week by combining the output of the freshly trained seasonal model and the residual model, clamping the final prediction at a minimum of zero using `max(0, pred_seasonal + pred_resid)`.
* **Error Quantification:** Once all 52 simulated weeks are complete, it calculates three key metrics to quantify model drift and reliability:
    * **Mean Absolute Error (MAE):** Utilizes `sklearn.metrics` to calculate the absolute raw difference between the predictions and the actual recorded values.
    * **Mean Absolute Percentage Error (MAPE):** Calculates the percentage-based error. Crucially, it uses `np.maximum(np.array(actuals), 1)` in the denominator to prevent division-by-zero crashes if a health metric (like hospitalizations) temporarily drops to zero in a given week.
    * **Residual Logging:** Beyond aggregate scores, the script records the individual `actual_value - final_pred` delta for every step in the loop. This array of residuals is exported to the JSON payload, allowing the frontend dashboard to plot error distributions or histograms.
* **Backtesting:** The script tests its own accuracy by looking back over the last 52 weeks (or fewer if historical data is limited).
* **Error Metrics:** For each past week, it simulates making a prediction using only data available prior to that week, then compares its prediction to the actual outcome. It calculates the Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE) to quantify the model's reliability.

### 5. Execution and Output
* **Processing Loop:** The script loops through all five metrics defined in the configuration, running the fetching, forecasting, and evaluation functions for each target.
* **Data Compilation:** It compiles the historical data, the 52-week forecast (with upper and lower bounds), accuracy metrics, and feature importance scores into a large Python dictionary.
* **JSON Export:** Finally, it exports this comprehensive dataset into a local file named `dashboard_data.json`.
