import sys
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Add the parent directory to system path for module imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# Import data loading and exploration functions
from data_exploration.data_loader import load_and_clean_data
from data_exploration.data_exploration import perform_data_exploration

# Define the base path
base_path = os.getcwd()

# Specify the file path (can be modified to load different files)
input_file_path = os.path.join(base_path, "water_data", "input", "water_consumption_2015_2023.csv")

# Load and clean the data using the data loader
df_cleaned = load_and_clean_data(input_file_path)

# Perform data exploration (optional)
perform_data_exploration(df_cleaned)

# Set the index to a period (daily)
df_cleaned.index = pd.DatetimeIndex(df_cleaned.index).to_period('D')

# Define the target and features
endog = df_cleaned['Wasserverbrauch']  # Target variable (water consumption)
exog = df_cleaned.drop(columns=['Wasserverbrauch'])  # Features (other variables)

# Clean the exogenous variables by replacing infinities and dropping NaN rows
exog_clean = exog.replace([np.inf, -np.inf], np.nan)  # Replace infinities with NaN
exog_clean = exog_clean.dropna()  # Drop rows with NaN values

# Ensure the target variable has no missing values for the cleaned exogenous variables
endog_clean = endog[exog_clean.index]

# Fit the ARIMA model (Auto-Regressive Integrated Moving Average)
# The model order is (1, 1, 1) for AR, I, MA respectively
model = sm.tsa.ARIMA(endog_clean, order=(1, 1, 1))

# Fit the model to the data
results = model.fit()

# Print the model summary
print(results.summary())

# Plot the diagnostics to check residuals and model fit
results.plot_diagnostics(figsize=(15, 10))
plt.show()

# Forecast the next 365 days (one year) with the given exogenous variables
forecast = results.get_forecast(steps=365, exog=exog_clean[-365:])  # Adjust forecast horizon
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Convert the index back to a DatetimeIndex for plotting
df_cleaned.index = df_cleaned.index.to_timestamp()

# Plot the actual vs forecasted data
plt.figure(figsize=(10, 6))
plt.plot(df_cleaned.index, df_cleaned['Wasserverbrauch'], label='Observed')  # Observed values
plt.plot(forecast_mean.index, forecast_mean, label='Forecast', color='red')  # Forecasted values
plt.fill_between(forecast_mean.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)  # Confidence interval
plt.legend()
plt.show()
