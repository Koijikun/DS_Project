import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from data_exploration.exploration import df_reduced
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Assuming df is your cleaned and pre-processed DataFrame
# Your target variable is 'Wasserverbrauch' (water consumption)
# You have dummy variables for weekdays ('is_monday', 'is_tuesday', etc.)

df_reduced.index = pd.DatetimeIndex(df_reduced.index).to_period('D')

# Step 1: Specify the endogenous and exogenous variables
endog = df_reduced['Wasserverbrauch']  # Endogenous variable (water consumption)
exog = df_reduced.drop(columns=['Wasserverbrauch'])  # Exogenous variables (weekday dummies)

# Remove rows with NaN or Inf in exogenous variables
exog_clean = exog.replace([np.inf, -np.inf], np.nan)  # Replace infinities with NaN
exog_clean = exog_clean.dropna()  # Drop rows with NaN values

# Ensure that the endogenous variable (target) also has no missing values
endog_clean = endog[exog_clean.index]


# Step 2: Fit the SARIMA model
# Seasonal order (P, D, Q, m) where m = 365 for yearly seasonality
# p, d, q are the non-seasonal AR, I, MA orders respectively
# Use only a subset of data for faster testing
endog_clean_subset = endog_clean[:365]  # First year of data
exog_clean_subset = exog_clean[:365]
model = sm.tsa.SARIMAX(endog_clean_subset, exog=exog_clean_subset,
                       order=(1, 1, 1),  # Non-seasonal AR, I, MA orders
                       seasonal_order=(1, 1, 1, 30),  # Seasonal AR, I, MA orders with yearly seasonality
                       enforce_stationarity=False,
                       enforce_invertibility=False)
results = model.fit(disp=False)

# Step 3: Fit the model
results = model.fit()

# Step 4: Display the results summary
print(results.summary())

# Step 5: Plot diagnostics to check the residuals
results.plot_diagnostics(figsize=(15, 10))
plt.show()

# Step 6: Forecasting (optional)
# Let's predict for the next 12 months (for example) or use any time range
forecast = results.get_forecast(steps=365, exog=exog[-365:])  # Adjust the forecast horizon
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Convert the period index to a datetime index for plotting
df_reduced.index = df_reduced.index.to_timestamp()

# Plot the observed data and the forecast
plt.figure(figsize=(10, 6))
plt.plot(df_reduced.index, df_reduced['Wasserverbrauch'], label='Observed')  # Now using DatetimeIndex
plt.plot(forecast_mean.index, forecast_mean, label='Forecast', color='red')
plt.fill_between(forecast_mean.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.legend()
plt.show()


