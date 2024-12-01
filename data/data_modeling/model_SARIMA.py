import sys
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Add parent directory to system path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# Import the data loading function, exploration functions, and plot functions
from data_exploration.data_loader import load_and_clean_data
from data_exploration.data_exploration import perform_data_exploration
from data_exploration.plot_functions import plot_forecast

# Define the base path
base_path = os.getcwd()

# Specify the file path (can be modified to load different files)
input_file_path = os.path.join(base_path, "water_data", "input" , "water_consumption_2015_2023.csv")


# Load and clean the data
df_cleaned = load_and_clean_data(input_file_path)

# Perform data exploration
perform_data_exploration(df_cleaned)

# Set the index to a period (daily) for time series analysis
df_cleaned.index = pd.DatetimeIndex(df_cleaned.index).to_period('D')

# Define target and features
endog = df_cleaned['Wasserverbrauch']  # Target variable (water consumption)
exog = df_cleaned.drop(columns=['Wasserverbrauch'])  # Features (weekday dummies)

# Clean the exogenous variables by replacing infinities and dropping NaN rows
exog_clean = exog.replace([np.inf, -np.inf], np.nan)  # Replace infinities with NaN
exog_clean = exog_clean.dropna()  # Drop rows with NaN values

# Ensure target variable has no missing values
endog_clean = endog[exog_clean.index]

# Fit the SARIMA model
model = sm.tsa.SARIMAX(endog_clean, exog=exog_clean,
                       order=(1, 1, 1),  # Non-seasonal AR, I, MA orders
                       seasonal_order=(1, 1, 1, 12),  # Seasonal AR, I, MA orders with yearly seasonality
                       enforce_stationarity=False,
                       enforce_invertibility=False)
results = model.fit(disp=False)

# Display the results summary
print(results.summary())

# Plot diagnostics to check the residuals
results.plot_diagnostics(figsize=(15, 10))
plt.show()

# Forecast for the next 365 days
forecast = results.get_forecast(steps=365, exog=exog[-365:])
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Convert PeriodIndex to DateTimeIndex for proper plotting
df_cleaned.index = df_cleaned.index.to_timestamp()

# Plot the observed data and the forecast using the plot_forecast function
plot_forecast(df_cleaned, forecast_mean, forecast_ci, observed_column='Wasserverbrauch')
