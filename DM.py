import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout
from keras.optimizers import Adam

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

# Split the data into training, validation, and testing sets
train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.1)
X_train, X_val, X_test = X[:train_size], X[train_size:train_size+val_size], X[train_size+val_size:]
y_train_1, y_val_1, y_test_1 = y_1[:train_size], y_1[train_size:train_size+val_size], y_1[train_size+val_size:]
y_train_5, y_val_5, y_test_5 = y_5[:train_size], y_5[train_size:train_size+val_size], y_5[train_size+val_size:]
y_train_22, y_val_22, y_test_22 = y_22[:train_size], y_22[train_size:train_size+val_size], y_22[train_size+val_size:]

# Combine train and validation sets for ARIMAX
X_train_arimax = pd.concat([X_train, X_val])
y_train_1_arimax = pd.concat([y_train_1, y_val_1])
y_train_5_arimax = pd.concat([y_train_5, y_val_5])
y_train_22_arimax = pd.concat([y_train_22, y_val_22])

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
model_1 = fit_arimax(X_train_arimax, y_train_1_arimax)
pred_1 = forecast_arimax(model_1, X_test)

model_5 = fit_arimax(X_train_arimax, y_train_5_arimax)
pred_5 = forecast_arimax(model_5, X_test)

model_22 = fit_arimax(X_train_arimax, y_train_22_arimax)
pred_22 = forecast_arimax(model_22, X_test)

# MinMaxScaler for neural networks
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Function to build and compile the DNN model
def build_dnn(input_shape, layers, lr, do):
    model = Sequential()
    model.add(Dense(layers[0], input_dim=input_shape, activation='relu'))
    for units in layers[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(d)) 
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    return model

def build_gru(input_shape, layers, lr, do):
    model = Sequential()
    model.add(GRU(layers[0], input_shape=(1, input_shape), return_sequences=True))
    for units in layers[1:-1]:
        model.add(GRU(units, return_sequences=True))
        model.add(Dropout(d))  # Add dropout to prevent overfitting
    model.add(GRU(layers[-1]))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    return model

# Function to build and compile the LSTM model
def build_lstm(input_shape, layers, lr, do):
    model = Sequential()
    model.add(LSTM(layers[0], input_shape=(1, input_shape), return_sequences=True))
    for units in layers[1:-1]:
        model.add(LSTM(units, return_sequences=True))
        model.add(Dropout(d))  # Add dropout to prevent overfitting
    model.add(LSTM(layers[-1]))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    return model

# Function to train and forecast with neural networks
def train_and_forecast_nn(model, X_train, y_train, X_val, y_val, X_test, epochs=50, batch_size=32):
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0)
    return model.predict(X_test)

# Calculate performance metrics
def calculate_metrics(y_true, y_pred, y_train):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    y_train_mean = np.mean(y_train)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    r2 = 1 - (np.sum((y_true - y_pred) ** 2))/(np.sum((y_true - y_train_mean) ** 2))
    
    smape = 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
    return rmse, r2, smape

# Reshape input for GRU and LSTM
X_train_nn = np.expand_dims(X_train_scaled, axis=1)
X_val_nn = np.expand_dims(X_val_scaled, axis=1)
X_test_nn = np.expand_dims(X_test_scaled, axis=1)

resultlist = []

hlist = [128,64,32]
llist = [0.001,0.0001]
dlist = [0,0.2]
for h in hlist:
    for l in llist:
        for d in dlist:
            # 1-day horizon models
            dnn_1 = build_dnn(X_train_scaled.shape[1], [h, int(h/2), int(h/4)], lr=l, do = d)
            gru_1 = build_gru(X_train_scaled.shape[1], [h], lr=l,do = d)
            lstm_1 = build_lstm(X_train_scaled.shape[1], [h], lr=l,do = d)
            
            pred_dnn_1 = train_and_forecast_nn(dnn_1, X_train_scaled, y_train_1, X_val_scaled, y_val_1, X_test_scaled)
            pred_gru_1 = train_and_forecast_nn(gru_1, X_train_nn, y_train_1, X_val_nn, y_val_1, X_test_nn)
            pred_lstm_1 = train_and_forecast_nn(lstm_1, X_train_nn, y_train_1, X_val_nn, y_val_1, X_test_nn)
            
            # 5-day horizon models
            dnn_5 = build_dnn(X_train_scaled.shape[1], [h, int(h/2), int(h/4)], lr=l,do = d)
            gru_5 = build_gru(X_train_scaled.shape[1], [h], lr=l,do = d)
            lstm_5 = build_lstm(X_train_scaled.shape[1], [h], lr=l,do = d)
            
            pred_dnn_5 = train_and_forecast_nn(dnn_5, X_train_scaled, y_train_5, X_val_scaled, y_val_5, X_test_scaled)
            pred_gru_5 = train_and_forecast_nn(gru_5, X_train_nn, y_train_5, X_val_nn, y_val_5, X_test_nn)
            pred_lstm_5 = train_and_forecast_nn(lstm_5, X_train_nn, y_train_5, X_val_nn, y_val_5, X_test_nn)
            
            # 22-day horizon models
            dnn_22 = build_dnn(X_train_scaled.shape[1], [h, int(h/2), int(h/4)], lr=l,do = d)
            gru_22 = build_gru(X_train_scaled.shape[1], [h], lr=l,do = d)
            lstm_22 = build_lstm(X_train_scaled.shape[1], [h], lr=l,do = d)
            
            pred_dnn_22 = train_and_forecast_nn(dnn_22, X_train_scaled, y_train_22, X_val_scaled, y_val_22, X_test_scaled)
            pred_gru_22 = train_and_forecast_nn(gru_22, X_train_nn, y_train_22, X_val_nn, y_val_22, X_test_nn)
            pred_lstm_22 = train_and_forecast_nn(lstm_22, X_train_nn, y_train_22, X_val_nn, y_val_22, X_test_nn)
            
            
            ## flatten the predictions
            
            pred_dnn_1 = [x for xs in pred_dnn_1 for x in xs]
            pred_dnn_5 = [x for xs in pred_dnn_5 for x in xs]
            pred_dnn_22 = [x for xs in pred_dnn_22 for x in xs]
            
            pred_gru_1 = [x for xs in pred_gru_1 for x in xs]
            pred_gru_5 = [x for xs in pred_gru_5 for x in xs]
            pred_gru_22 = [x for xs in pred_gru_22 for x in xs]
            
            pred_lstm_1 = [x for xs in pred_lstm_1 for x in xs]
            pred_lstm_5 = [x for xs in pred_lstm_5 for x in xs]
            pred_lstm_22 = [x for xs in pred_lstm_22 for x in xs]
            
            
            ## DNN
            rmse_1_dnn, r2_1_dnn, smape_1_dnn = calculate_metrics(y_test_1, pred_dnn_1, y_train_1)
            rmse_5_dnn, r2_5_dnn, smape_5_dnn = calculate_metrics(y_test_5, pred_dnn_5, y_train_5)
            rmse_22_dnn, r2_22_dnn, smape_22_dnn = calculate_metrics(y_test_22, pred_dnn_22, y_train_22)
            
            resultlist.append(("DNN_1", str([h, int(h/2), int(h/4)]), l, d, rmse_1_dnn, r2_1_dnn, smape_1_dnn))
            resultlist.append(("DNN_5", str([h, int(h/2), int(h/4)]), l, d, rmse_5_dnn, r2_5_dnn, smape_5_dnn))
            resultlist.append(("DNN_22", str([h, int(h/2), int(h/4)]), l, d, rmse_22_dnn, r2_22_dnn, smape_22_dnn))
            
            ## LSTM
            rmse_1_lstm, r2_1_lstm, smape_1_lstm = calculate_metrics(y_test_1, pred_lstm_1, y_train_1)
            rmse_5_lstm, r2_5_lstm, smape_5_lstm = calculate_metrics(y_test_5, pred_lstm_5, y_train_5)
            rmse_22_lstm, r2_22_lstm, smape_22_lstm = calculate_metrics(y_test_22, pred_lstm_22, y_train_22)
            
            resultlist.append(("LSTM_1", h, l, d, rmse_1_lstm, r2_1_lstm, smape_1_lstm))
            resultlist.append(("LSTM_5", h, l, d, rmse_5_lstm, r2_5_lstm, smape_5_lstm))
            resultlist.append(("LSTM_22", h, l, d, rmse_22_lstm, r2_22_lstm, smape_22_lstm))


            ## GRU
            rmse_1_gru, r2_1_gru, smape_1_gru = calculate_metrics(y_test_1, pred_gru_1, y_train_1)
            rmse_5_gru, r2_5_gru, smape_5_gru = calculate_metrics(y_test_5, pred_gru_5, y_train_5)
            rmse_22_gru, r2_22_gru, smape_22_gru = calculate_metrics(y_test_22, pred_gru_22, y_train_22)
            
            resultlist.append(("GRU_1", h, l, d, rmse_1_gru, r2_1_gru, smape_1_gru))
            resultlist.append(("GRU_5", h, l, d, rmse_5_gru, r2_5_gru, smape_5_gru))
            resultlist.append(("GRU_22", h, l, d, rmse_22_gru, r2_22_gru, smape_22_gru))

            
            

## plotting the best models 



dnn_1 = build_dnn(X_train_scaled.shape[1], [128, 64, 32], lr=0.001, do = 0.2)
gru_1 = build_gru(X_train_scaled.shape[1],  [128], lr=0.001 ,do = 0.2)
lstm_1 = build_lstm(X_train_scaled.shape[1], [128], lr=0.001,do = 0)

pred_dnn_1 = train_and_forecast_nn(dnn_1, X_train_scaled, y_train_1, X_val_scaled, y_val_1, X_test_scaled)
pred_gru_1 = train_and_forecast_nn(gru_1, X_train_nn, y_train_1, X_val_nn, y_val_1, X_test_nn)
pred_lstm_1 = train_and_forecast_nn(lstm_1, X_train_nn, y_train_1, X_val_nn, y_val_1, X_test_nn)

# 5-day horizon models
dnn_5 = build_dnn(X_train_scaled.shape[1], [64, 32, 16], lr=0.0001 ,do = 0)
gru_5 = build_gru(X_train_scaled.shape[1], [128], lr=0.001 ,do = 0.2)
lstm_5 = build_lstm(X_train_scaled.shape[1], [128], lr=0.001, do = 0)

pred_dnn_5 = train_and_forecast_nn(dnn_5, X_train_scaled, y_train_5, X_val_scaled, y_val_5, X_test_scaled)
pred_gru_5 = train_and_forecast_nn(gru_5, X_train_nn, y_train_5, X_val_nn, y_val_5, X_test_nn)
pred_lstm_5 = train_and_forecast_nn(lstm_5, X_train_nn, y_train_5, X_val_nn, y_val_5, X_test_nn)

# 22-day horizon models
dnn_22 = build_dnn(X_train_scaled.shape[1], [32, 16, 8], lr=0.0001,do = 0.2)
gru_22 = build_gru(X_train_scaled.shape[1], [64], lr=0.001,do = 0.2)
lstm_22 = build_lstm(X_train_scaled.shape[1], [32], lr=0.001,do = 0.2)

pred_dnn_22 = train_and_forecast_nn(dnn_22, X_train_scaled, y_train_22, X_val_scaled, y_val_22, X_test_scaled)
pred_gru_22 = train_and_forecast_nn(gru_22, X_train_nn, y_train_22, X_val_nn, y_val_22, X_test_nn)
pred_lstm_22 = train_and_forecast_nn(lstm_22, X_train_nn, y_train_22, X_val_nn, y_val_22, X_test_nn)



# Plot forecasts
def plot_forecasts(y_true, y_preds, horizon, labels):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.index, y_true, label='Actual', color='black')
    for y_pred, label in zip(y_preds, labels):
        plt.plot(y_true.index, y_pred, label=label)
    plt.title(f'{horizon}-day Ahead Forecast')
    plt.xlabel('Time')
    plt.ylabel('Nord Pool Spot Price')
    plt.legend()
    plt.show()

# Plot for 1-day ahead
plot_forecasts(y_test_1, [pred_dnn_1, pred_gru_1, pred_lstm_1, pred_1], 1, ['DNN', 'GRU', 'LSTM', 'ARIMAX'])

# Plot for 5-day ahead
plot_forecasts(y_test_5, [pred_dnn_5, pred_gru_5, pred_lstm_5, pred_5], 5, ['DNN', 'GRU', 'LSTM', 'ARIMAX'])

# Plot for 22-day ahead
plot_forecasts(y_test_22, [pred_dnn_22, pred_gru_22, pred_lstm_22, pred_22], 22, ['DNN', 'GRU', 'LSTM', 'ARIMAX'])

#############


# Fit and forecast for each horizon with multiple ARIMAX orders
orders = [(1, 1, 1), (1, 1, 2), (1, 2, 1), (2, 1, 1), (2, 2, 1), (2, 1, 2), (1, 2, 2), (2, 2, 2)]

# Prepare a dictionary to store the results
results = {'Horizon 1': [], 'Horizon 5': [], 'Horizon 22': []}

# Define horizons
horizons = {'Horizon 1': (X_train_arimax, y_train_1_arimax, y_test_1),
            'Horizon 5': (X_train_arimax, y_train_5_arimax, y_test_5),
            'Horizon 22': (X_train_arimax, y_train_22_arimax, y_test_22)}

# Loop through each horizon and order
for horizon, (X_train, y_train, y_test) in horizons.items():
    for order in orders:
        model_fit = fit_arimax(X_train, y_train, order)
        pred = forecast_arimax(model_fit, X_test)
        rmse, oosr2, smape = calculate_metrics(y_test, pred, y_train)
        results[horizon].append((order, rmse, oosr2, smape))

# Print results
for horizon, metrics in results.items():
    print(f"\n{horizon}:")
    for order, rmse, oosr2, smape in metrics:
        print(f"Order: {order}, RMSE: {rmse:.4f}, OOSR2: {oosr2:.4f}, SMAPE: {smape:.4f}")
        
##############

model_1 = fit_arimax(X_train_arimax, y_train_1_arimax, (2, 1, 2))
pred_1 = forecast_arimax(model_1, X_test)

model_5 = fit_arimax(X_train_arimax, y_train_5_arimax, (2, 1, 1))
pred_5 = forecast_arimax(model_5, X_test)

model_22 = fit_arimax(X_train_arimax, y_train_22_arimax, (2, 1, 2))
pred_22 = forecast_arimax(model_22, X_test)


# Plot for 1-day ahead
plot_forecasts(y_test_1, [pred_dnn_1, pred_gru_1, pred_lstm_1, pred_1], 1, ['DNN', 'GRU', 'LSTM', 'ARIMAX'])

# Plot for 5-day ahead
plot_forecasts(y_test_5, [pred_dnn_5, pred_gru_5, pred_lstm_5, pred_5], 5, ['DNN', 'GRU', 'LSTM', 'ARIMAX'])

# Plot for 22-day ahead
plot_forecasts(y_test_22, [pred_dnn_22, pred_gru_22, pred_lstm_22, pred_22], 22, ['DNN', 'GRU', 'LSTM', 'ARIMAX'])

# Plot forecasts with date as label and correct axis. delete the other one when you have time 
def plot_forecasts(y_true, y_preds, horizon, labels, dates):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_true, label='Actual', color='black')
    for y_pred, label in zip(y_preds, labels):
        plt.plot(dates, y_pred, label=label)
    plt.title(f'{horizon}-day Ahead Forecast Nord Pool Spot Price')
    plt.xlabel('Time')
    plt.ylabel('Euros/MWh')
    plt.legend()
    plt.show()
    
    
dates = df['Date'].tail(799)

# Plot for 1-day ahead
plot_forecasts(y_test_1, [pred_dnn_1, pred_gru_1, pred_lstm_1, pred_1], 1, ['DNN', 'GRU', 'LSTM', 'ARIMAX'], dates)

# Plot for 5-day ahead
plot_forecasts(y_test_5, [pred_dnn_5, pred_gru_5, pred_lstm_5, pred_5], 5, ['DNN', 'GRU', 'LSTM', 'ARIMAX'], dates)

# Plot for 22-day ahead
plot_forecasts(y_test_22, [pred_dnn_22, pred_gru_22, pred_lstm_22, pred_22], 22, ['DNN', 'GRU', 'LSTM', 'ARIMAX'], dates)



############################# DM ##############################

from statsmodels.tsa.stattools import acf
from scipy.stats import norm


def dm_test(e1, e2, h=1):
    d = np.array(e1)**2 - np.array(e2)**2
    mean_d = np.mean(d)
    autocov_d = acf(d, fft=True, nlags=h)
    var_d = np.var(d, ddof=1) + 2 * np.sum(autocov_d[1:])

    dm_stat = mean_d / np.sqrt(var_d / len(d))
    p_value = 2 * norm.cdf(-np.abs(dm_stat))
    
    return dm_stat, p_value

pred_dnn_1 = pred_dnn_1.reshape(-1)
pred_lstm_1 = pred_lstm_1.reshape(-1)
pred_gru_1 = pred_gru_1.reshape(-1)
pred_arimax_1 = pred_1.to_numpy().reshape(-1)

np_y_test = y_test.to_numpy().reshape(-1)

forecast_errors = {
    'dnn_1': [a_i - b_i for a_i, b_i in zip(pred_dnn_1, np_y_test)],
    'lstm_1':  [a_i - b_i for a_i, b_i in zip(pred_lstm_1, np_y_test)],
    'gru_1':  [a_i - b_i for a_i, b_i in zip(pred_gru_1, np_y_test)],
    'arimax_1':  [a_i - b_i for a_i, b_i in zip(pred_1, np_y_test)],
}


models = list(forecast_errors.keys())
num_models = len(models)
conf_matrix = np.zeros((num_models, num_models))

for i in range(num_models):
    for j in range(num_models):
        if i != j:
            e1 = forecast_errors[models[i]]
            e2 = forecast_errors[models[j]]
            _, p_value = dm_test(e1, e2)
            if p_value < 0.5:  # Model i significantly outperforms Model j
                conf_matrix[i, j] = 1
            elif p_value >= 0.5:  # Model j significantly outperforms Model i
                conf_matrix[j, i] = 1

# Calculate the percentage of outperformance
for i in range(num_models):
    for j in range(num_models):
        if i != j:
            total_comparisons = sum(conf_matrix[i]) + sum(conf_matrix[j])
            if total_comparisons > 0:
                conf_matrix[i, j] = (conf_matrix[i, j] / total_comparisons) * 100

conf_df = pd.DataFrame(conf_matrix, index=models, columns=models).astype(int).astype(str) + '%'
np.fill_diagonal(conf_df.values, '-')
print(conf_df)

































