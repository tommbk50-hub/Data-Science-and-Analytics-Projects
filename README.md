# Influenza and COVID case prediction using machine learning

This dashboard predicts future trends using a HistGradientBoostingRegressor trained on historical surveillance data. The system is fully automated: it retrieves live data directly from the UKHSA public API and retrains the model every week via GitHub Actions, ensuring the forecast is always based on the latest available statistics.

This dashboard tracks and predicts the percentage of people who received a PCR test for Influenza (flu) and had at least one positive test result in the same 7 days. This dashboard predicts future trends using a HistGradientBoostingRegressor (SciKitLearn) trained on historical surveillance data from the UK Health Security Agency. 

The system is fully automated: it retrieves live data directly from the UKHSA public API and retrains the model every week via GitHub Actions, ensuring the forecast is always based on the latest available statistics.

Data is shown by specimen date (the date the sample was collected).

