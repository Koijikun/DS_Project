import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from data_exploration.exploration import df_reduced

from data_exploration.exploration import df_reduced
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Create lag features manually (e.g., for 7 days)
df_reduced['lag_1'] = df_reduced['Wasserverbrauch'].shift(1)
df_reduced['lag_2'] = df_reduced['Wasserverbrauch'].shift(2)
df_reduced['lag_3'] = df_reduced['Wasserverbrauch'].shift(3)
df_reduced['lag_4'] = df_reduced['Wasserverbrauch'].shift(4)
df_reduced['lag_5'] = df_reduced['Wasserverbrauch'].shift(5)
df_reduced['lag_6'] = df_reduced['Wasserverbrauch'].shift(6)
df_reduced['lag_7'] = df_reduced['Wasserverbrauch'].shift(7)

# Additional rolling statistics
df_reduced['rolling_mean_3'] = df_reduced['Wasserverbrauch'].rolling(window=3).mean()
df_reduced['rolling_mean_7'] = df_reduced['Wasserverbrauch'].rolling(window=7).mean()
df_reduced['rolling_std_3'] = df_reduced['Wasserverbrauch'].rolling(window=3).std()

# Convert Saturday and Sunday dummies to integers (only these two days)
df_reduced['is_saturday'] = df_reduced['is_saturday'].astype(int)
df_reduced['is_sunday'] = df_reduced['is_sunday'].astype(int)

# Add more features (e.g., month, weekday)
df_reduced['month'] = df_reduced.index.month
df_reduced['weekday'] = df_reduced.index.weekday

# Drop rows with NaN values that were created by shifting
df_lagged = df_reduced.dropna()

# Step 2: Prepare the data for training
X = df_lagged[['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'lag_6', 'lag_7', 
               'is_saturday', 'is_sunday', 'month', 'weekday', 
               'rolling_mean_3', 'rolling_mean_7', 'rolling_std_3']]  # More features
y = df_lagged['Wasserverbrauch']  # Endogenous variable (target)

# Step 3: TimeSeries Split for Cross-Validation (better for time series)
tscv = TimeSeriesSplit(n_splits=5)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 5: Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf_model = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=tscv, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best model after hyperparameter tuning
best_rf_model = grid_search.best_estimator_

# Step 6: Make predictions on the test set
y_pred = best_rf_model.predict(X_test)

# Step 7: Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the model evaluation metrics
print(f'Mean Squared Error (MSE): {mse}')
print(f'Mean Absolute Error (MAE): {mae}')
print(f'R-squared (R²): {r2}')

# Step 8: Print the actual vs predicted values (first few rows)
print("\nActual vs Predicted Values:")
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison.head())

# Step 9: Plot the observed vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Observed')
plt.plot(y_test.index, y_pred, label='Predicted', color='red')
plt.legend()
plt.title('Random Forest Predictions vs Observed Values')
plt.show()

# Step 10: Plot residuals (Actual - Predicted)
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, residuals, label='Residuals', color='purple')
plt.axhline(y=0, color='black', linestyle='--')
plt.legend()
plt.title('Residuals (Observed - Predicted)')
plt.show()

# Step 11: Feature importance
importances = best_rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), features[indices])
plt.xlabel("Relative Importance")
plt.show()
