import matplotlib.pyplot as plt
import load_clean_data as ld
import plot_functions as pl
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# Visualize data in basic plot
pl.plot_aggregated_data(ld.df, days=1, column='Wasserverbrauch')
# Data shows a clear structural break during 2020 which has to be handled later

# Show moving average
ld.df['rolling_mean'] = ld.df['Wasserverbrauch'].rolling(window=30).mean()
plt.plot(ld.df.index, ld.df['Wasserverbrauch'], label='Original')
plt.plot(ld.df.index, ld.df['rolling_mean'], label='30-Day Rolling Mean', color='red')
plt.legend()
plt.show()

# Check correlation --> which variable to keep/eliminate
sns.heatmap(ld.df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.show()
# Most important features: Geburte, RainDur_min, StrGlo_W/m2, T_C, rolling_mean
# Eliminate due to multicollinearity: T_max_h1_C
# Keep watching: Multicollinearity for StrGlo_W/m2 & T_C: 0.74

# stationarity check --> smaller than 0.05 = stationary
result = adfuller(ld.df['Wasserverbrauch'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')
# Data is stationary

# ACF & PACF
plot_acf(ld.df['Wasserverbrauch'].dropna(), lags=100)
plot_pacf(ld.df['Wasserverbrauch'].dropna(), lags=100)
plt.show()
# As expected seasonality is present
# +/- 10 significant lags

# dropping 'insignificant' features
df_reduced = ld.df.drop(columns=['Veränderung Vortag','Wegzüge','Zuzüge','Todesfälle','T_max_h1_C','p_hPa'])

# adding previous values to help with forecasting at a later point
df_reduced['lag_1'] = df_reduced['Wasserverbrauch'].shift(1)
df_reduced['lag_2'] = df_reduced['Wasserverbrauch'].shift(365)
# pending: fill values where lag isnt possible



sns.heatmap(df_reduced.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

