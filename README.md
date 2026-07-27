# Influenza and COVID case prediction using machine learning

This [dashboard](https://tommbk50-hub.github.io/UK-respiratory-illness-case-predictions-using-machine-learning/dashboard.html) uses a Hybrid Machine Learning strategy (HistGradientBoostingRegressor) trained on historical surveillance data from the [UK Health Security Agency website](https://ukhsa-dashboard.data.gov.uk/) to provide a comprehensive 52-week forecast for Influenza (Flu) and COVID-19 in England using machine learning to predict future patterns. By analysing trends in PCR Positivity and Hospital Admissions, these models aim to support healthcare resource planning—helping hospitals and A&E departments anticipate potential winter surges.

The system is fully automated, retrieving live data from the UKHSA public API each week, then iteratively retraining the model every week via GitHub Actions, ensuring the forecast is always based on the latest available statistics.

