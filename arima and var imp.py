import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score

# Load and prepare the data
df = pd.read_excel('C:/Users/jstam/Home/Study/UVT/Thesis\Data/all_data/thesis_data_JAS_final.xlsx')
df = df.fillna(method='ffill')

# Create lagged features
df['spot_lag_1'] = df['norden_spot'].shift(1)
df['spot_lag_7'] = df['norden_spot'].shift(5)
df['spot_lag_30'] = df['norden_spot'].shift(22)
df['spot_lag_90'] = df['norden_spot'].shift(66)
df['spot_lag_250'] = df['norden_spot'].shift(250)

# Create momentum features
df['mom_7'] = (df['spot_lag_1'] - df['spot_lag_7']) / df['spot_lag_7']
df['mom_30'] = (df['spot_lag_1'] - df['spot_lag_30']) / df['spot_lag_30']
df['mom_90'] = (df['spot_lag_1'] - df['spot_lag_90']) / df['spot_lag_90']
df['mom_250'] = (df['spot_lag_1'] - df['spot_lag_250']) / df['spot_lag_250']
df = df.dropna().reset_index(drop=True)

# Add season dummies
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'

df['Season'] = df['Month'].apply(get_season)
season_dummies = pd.get_dummies(df['Season'], prefix='Season')
season_dummies = season_dummies.astype(int)
df = pd.concat([df, season_dummies], axis=1)
df.drop(['Month', 'Season'], axis=1, inplace=True)

# Create future target columns
df['norden_spot_1'] = df['norden_spot'].shift(-1)
df['norden_spot_5'] = df['norden_spot'].shift(-5)
df['norden_spot_22'] = df['norden_spot'].shift(-22)
df = df.dropna().reset_index(drop=True)

# Prepare the data for each forecast horizon
y_1 = df['norden_spot_1']
y_5 = df['norden_spot_5']
y_22 = df['norden_spot_22']
X = df.drop(['norden_spot', 'norden_spot_1', 'norden_spot_5', 'norden_spot_22', 'Date'], axis=1)

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train_1, y_test_1 = y_1[:train_size], y_1[train_size:]
y_train_5, y_test_5 = y_5[:train_size], y_5[train_size:]
y_train_22, y_test_22 = y_22[:train_size], y_22[train_size:]

# Define and fit the ARIMAX model
def fit_arimax(X_train, y_train, order=(1, 2, 2)):
    model = SARIMAX(y_train, exog=X_train, order=order)
    model_fit = model.fit(disp=False)
    return model_fit

# Make forecasts
def forecast_arimax(model_fit, X_test):
    forecast = model_fit.get_forecast(steps=len(X_test), exog=X_test)
    return forecast.predicted_mean

# Fit and forecast for each horizon
model_1 = fit_arimax(X_train, y_train_1)
pred_1 = forecast_arimax(model_1, X_test)

model_5 = fit_arimax(X_train, y_train_5)
pred_5 = forecast_arimax(model_5, X_test)

model_22 = fit_arimax(X_train, y_train_22)
pred_22 = forecast_arimax(model_22, X_test)

# Calculate performance metrics
def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    smape = 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    return rmse, r2, smape

rmse_1, r2_1, smape_1 = calculate_metrics(y_test_1, pred_1)
rmse_5, r2_5, smape_5 = calculate_metrics(y_test_5, pred_5)
rmse_22, r2_22, smape_22 = calculate_metrics(y_test_22, pred_22)


