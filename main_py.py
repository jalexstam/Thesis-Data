import numpy as np
import pandas as pd

np.random.seed(42)

df = pd.read_excel('thesis_data_JAS_final.xlsx')
df

## DNN
df= filtered_df.fillna(method='ffill')
df['spot_lag_1'] = df['norden_spot'].shift(1)
df['spot_lag_7'] = df['norden_spot'].shift(5) ## its called 7 to indicate week but its only 5 workdays
df['spot_lag_30'] = df['norden_spot'].shift(22) ## its called 30 to indicate month but its only 22 workdays
df['mom_30'] = (df['spot_lag_1']    -   df['spot_lag_30'])     /   df['spot_lag_30']
df = df.dropna().reset_index(drop=True)



df['Date'] = pd.to_datetime(tester['Date'])

# Extract the month from the 'Date' column
df['Month'] = df['Date'].dt.month

# Define a function to categorize months into seasons
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'

# Apply the function to create a new 'Season' column
df['Season'] = df['Month'].apply(get_season)

# Convert the 'Season' column to dummy variables
season_dummies = pd.get_dummies(df['Season'], prefix='Season')
season_dummies = season_dummies.astype(int)

# Concatenate the original DataFrame with the season dummies
df = pd.concat([tester, season_dummies], axis=1)

# Drop the 'Month' and 'Season' columns if no longer needed
df.drop(['Month', 'Season'], axis=1, inplace=True)


y = df['norden_spot']
X = df.iloc[:,2:]
X = sm.add_constant(X)
model = sm.OLS(y, X)
results = model.fit()
results.summary()