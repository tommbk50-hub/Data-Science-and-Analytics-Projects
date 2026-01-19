import pandas as pd
import numpy as np
import json
from sklearn.ensemble import HistGradientBoostingRegressor

# 1. Load Data
try:
    df = pd.read_csv('ukhsa-chart-download.csv')
except FileNotFoundError:
    print("Error: 'ukhsa-chart-download.csv' not found.")
    exit()

df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)
df.set_index('date', inplace=True)

# -------------------------------------------------------
# STEP A: Train Seasonal Model
# -------------------------------------------------------
df['Week_Number'] = df.index.isocalendar().week.astype(int)
df['Month'] = df.index.month

seasonal_features = ['Week_Number', 'Month']
target = 'metric_value'

model_seasonal = HistGradientBoostingRegressor(categorical_features=[0, 1], random_state=42)
model_seasonal.fit(df[seasonal_features], df[target])

df['Seasonal_Pred'] = model_seasonal.predict(df[seasonal_features])
df['Residual'] = df['metric_value'] - df['Seasonal_Pred']

# -------------------------------------------------------
# STEP B: Train Recursive Model on RESIDUALS
# -------------------------------------------------------
df['Res_Lag_1'] = df['Residual'].shift(1)
df['Res_Lag_2'] = df['Residual'].shift(2)
df['Res_Lag_3'] = df['Residual'].shift(3)

df_resid = df.dropna().copy()
resid_features = ['Res_Lag_1', 'Res_Lag_2', 'Res_Lag_3', 'Week_Number'] 
resid_target = 'Residual'

model_resid = HistGradientBoostingRegressor(random_state=42)
model_resid.fit(df_resid[resid_features], df_resid[resid_target])

# -------------------------------------------------------
# STEP C: Hybrid Forecasting Loop
# -------------------------------------------------------
last_date = df.index[-1]
history_residuals = df['Residual'].iloc[-3:].tolist()
current_date = last_date
future_forecasts = []

for i in range(52): # Forecast 1 year ahead
    current_date = current_date + pd.Timedelta(days=7)

    # 1. Seasonal Baseline
    feat_week = current_date.isocalendar().week
    feat_month = current_date.month
    seasonal_base = model_seasonal.predict(pd.DataFrame([[feat_week, feat_month]], columns=seasonal_features))[0]

    # 2. Predict Residual
    res_lag_1 = history_residuals[-1]
    res_lag_2 = history_residuals[-2]
    res_lag_3 = history_residuals[-3]
    pred_residual = model_resid.predict(pd.DataFrame([[res_lag_1, res_lag_2, res_lag_3, feat_week]], columns=resid_features))[0]

    # 3. Combine
    final_pred = seasonal_base + pred_residual

    # 4. Update history
    history_residuals.append(pred_residual)
    future_forecasts.append({
        'date': current_date.strftime('%Y-%m-%d'),
        'Seasonal_Base': float(seasonal_base),
        'Final_Forecast': float(final_pred) # Ensure it's not negative if desired: max(0, float(final_pred))
    })

# -------------------------------------------------------
# STEP D: Export Data for Web
# -------------------------------------------------------
# We create a dictionary containing lists for the frontend to plot
output_data = {
    "history": {
        "dates": df.index.strftime('%Y-%m-%d').tolist(),
        "values": df['metric_value'].tolist()
    },
    "forecast": {
        "dates": [x['date'] for x in future_forecasts],
        "values": [x['Final_Forecast'] for x in future_forecasts],
        "baseline": [x['Seasonal_Base'] for x in future_forecasts]
    }
}

with open('dashboard_data.json', 'w') as f:
    json.dump(output_data, f)

print("Success: 'dashboard_data.json' created.")