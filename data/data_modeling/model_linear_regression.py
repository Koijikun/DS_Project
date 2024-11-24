# subfolder/sub_script.py
import sys
import os

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from data_exploration.exploration import df_reduced
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Assuming df_reduced is your cleaned and pre-processed DataFrame
# Your target variable is 'Wasserverbrauch' (water consumption)
# You have dummy variables for weekdays ('is_monday', 'is_tuesday', etc.)

df_reduced.index = pd.DatetimeIndex(df_reduced.index).to_period('D')

# Combine Saturday and Sunday into one 'is_weekend' variable
df_reduced['is_weekend'] = df_reduced['is_saturday'] | df_reduced['is_sunday']  # Using bitwise OR to combine
df_reduced = df_reduced.drop(columns=['is_saturday', 'is_sunday'])  # Drop individual weekend columns

# Step 1: Specify the endogenous and exogenous variables
endog = df_reduced['Wasserverbrauch']  # Endogenous variable (water consumption)
exog = df_reduced.drop(columns=['Wasserverbrauch','rolling_mean','lag_2','RainDur_min','Geburte','StrGlo_W/m2'])  # Exogenous variables

# Add new features: squared terms and interaction terms
exog['Temp^2'] = df_reduced['T_C'] ** 2
exog['StrGlo^2'] = df_reduced['StrGlo_W/m2'] ** 2
exog['RainDur_min^2'] = df_reduced['RainDur_min'] ** 2
exog['RainDur_min*weekend'] = df_reduced['RainDur_min'] * df_reduced['is_weekend']  # Interaction term

# Remove rows with NaN or Inf in exogenous variables
exog_clean = exog.replace([np.inf, -np.inf], np.nan)  # Replace infinities with NaN
exog_clean = exog_clean.dropna()  # Drop rows with NaN values

# Ensure that the endogenous variable (target) also has no missing values
endog_clean = endog[exog_clean.index]

# Step 2: Add a constant to the exogenous variables for the intercept in the linear regression
exog_clean = sm.add_constant(exog_clean)

# Step 3: Split into train and test set
# Use the last 360 rows for the test set
train_size = len(exog_clean) - 360
train_exog = exog_clean.iloc[:train_size]
test_exog = exog_clean.iloc[train_size:]
train_endog = endog_clean.iloc[:train_size]
test_endog = endog_clean.iloc[train_size:]

# Step 4: Fit the Linear Regression model (OLS - Ordinary Least Squares) on the train set
model = sm.OLS(train_endog, train_exog)  # Fit the model using OLS
results = model.fit()

# Step 5: Predict on the test set
test_predictions = results.predict(test_exog)

# Step 6: Display the results summary
print(results.summary())

# Step 7: Forecasting (plotting test set predictions)
# Convert PeriodIndex to DateTimeIndex for proper plotting
df_reduced.index = df_reduced.index.to_timestamp()

# Now, align the observed and predicted values for the test set
observed_test_values = test_endog
predicted_test_values = test_predictions

# Convert PeriodIndex to DateTimeIndex for proper plotting
observed_test_values.index = observed_test_values.index.to_timestamp()
predicted_test_values.index = predicted_test_values.index.to_timestamp()

# Now you can plot the data (actual vs predicted)
plt.figure(figsize=(10, 6))
plt.plot(observed_test_values.index, observed_test_values, label='Observed', color='blue')
plt.plot(predicted_test_values.index, predicted_test_values, label='Predicted', color='red', linestyle='--')

# Add labels and title
plt.title('Actual vs Predicted Water Consumption (Test Set)')
plt.xlabel('Date')
plt.ylabel('Water Consumption')

# Show the legend
plt.legend()

# Show the plot
plt.show()

