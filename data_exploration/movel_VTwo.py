from exploration import df_reduced
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
model = sm.tsa.SARIMAX(endog_clean, exog=exog_clean,
                       order=(1, 1, 1),  # Non-seasonal AR, I, MA orders (can be adjusted)
                       seasonal_order=(1, 1, 1, 365),  # Seasonal AR, I, MA orders with yearly seasonality
                       enforce_stationarity=False,
                       enforce_invertibility=False)

# Step 3: Fit the model
results = model.fit(disp=False)

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

# Plot the forecast
plt.figure(figsize=(10, 6))
plt.plot(df_reduced.index, df_reduced['Wasserverbrauch'], label='Observed')
plt.plot(forecast_mean.index, forecast_mean, label='Forecast', color='red')
plt.fill_between(forecast_mean.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.legend()
plt.show()

