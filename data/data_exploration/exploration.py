import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import matplotlib.pyplot as plt
import data_exploration.load_clean_data as ld
import data_exploration.plot_functions as pl
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# plot aggregated data
def plot_aggregated_data(df, days=7, column='Wasserverbrauch'):
    pl.plot_aggregated_data(df, days=days, column=column)

# Function to plot the moving average with a given window size
def plot_moving_average(df, column='Wasserverbrauch', window=30):
    df['rolling_mean'] = df[column].rolling(window=window).mean()
    plt.figure(figsize=(10, 6))
    plt.plot(df.index, df[column], label='Original')
    plt.plot(df.index, df['rolling_mean'], label=f'{window}-Day Rolling Mean', color='red')
    plt.legend()
    plt.title(f'{column} with {window}-Day Rolling Mean')
    plt.show()

# Function to plot correlation matrix of the dataset
def plot_correlation_matrix(df, annot=True, cmap='coolwarm', fmt='.2f'):
    """
    Plot the correlation matrix to check for multicollinearity and feature relationships.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=annot, cmap=cmap, fmt=fmt)
    plt.title('Correlation Matrix')
    plt.show()

# Function to check stationarity using ADF Test
def check_stationarity(df, column='Wasserverbrauch', threshold=0.05):
    """
    Perform the Augmented Dickey-Fuller test to check if the data is stationary.
    """
    result = adfuller(df[column])
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] < threshold:
        print(f"The series is stationary (p-value < {threshold}).")
    else:
        print(f"The series is non-stationary (p-value >= {threshold}).")

# Function to plot ACF and PACF
def plot_acf_pacf(df, column='Wasserverbrauch', lags=100):
    plot_acf(df['Wasserverbrauch'].dropna(), lags=100)
    plot_pacf(df['Wasserverbrauch'].dropna(), lags=100)
    plt.show()

# Function to drop 'insignificant' features based on analysis
def drop_insignificant_features(df, columns_to_drop=None):
    """
    Drop columns that are considered insignificant for the analysis.
    """
    if columns_to_drop is None:
        columns_to_drop = ['Veränderung Vortag','Wegzüge','Zuzüge','Todesfälle','T_max_h1_C','p_hPa']
    return df.drop(columns=columns_to_drop)

# Function to add lag features for forecasting
def add_lag_features(df, column='Wasserverbrauch', lags=[1, 2]):
    """
    Add lag features for time series forecasting (e.g., lag_1, lag_2).
    """
    for lag in lags:
        df[f'lag_{lag}'] = df[column].shift(lag)
    return df


# 1. Visualize data in basic plot (aggregated data)
plot_aggregated_data(ld.df, days=7, column='Wasserverbrauch')
# Data shows a clear structural break during 2020 which has to be handled later

# 2. Show moving average with a 30-day window
plot_moving_average(ld.df, column='Wasserverbrauch', window=30)

# 3. Check correlation matrix and visualize it
plot_correlation_matrix(ld.df, annot=True, cmap='coolwarm', fmt='.2f')
# Most important features: Geburte, RainDur_min, StrGlo_W/m2, T_C, rolling_mean
# Eliminate due to multicollinearity: T_max_h1_C
# Keep watching: Multicollinearity for StrGlo_W/m2 & T_C: 0.74

# 4. Check stationarity of the time series data
check_stationarity(ld.df, column='Wasserverbrauch')

# 5. Plot ACF and PACF to check for seasonality
plot_acf_pacf(ld.df, column='Wasserverbrauch', lags=100)
# As expected seasonality is present
# +/- 10 significant lags

# 6. Drop 'insignificant' features
df_reduced = drop_insignificant_features(ld.df)

# 7. Add lag features to the dataset for forecasting
df_reduced = add_lag_features(df_reduced, column='Wasserverbrauch', lags=[1, 2])

# 8. Visualize the correlation matrix of the reduced dataset
plot_correlation_matrix(df_reduced, annot=True, cmap='coolwarm', fmt='.2f')
