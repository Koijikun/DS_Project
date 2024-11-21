import pandas as pd

# Load your CSV data with the correct date format
df = pd.read_csv('./water_consumption_2015_2023.csv', delimiter=';', parse_dates=['Datum'], index_col='Datum', dayfirst=True)

# Display the first few rows to inspect the data
print(df.head())

# Calculate IQR and filter out outliers
Q1 = df['Wasserverbrauch'].quantile(0.25)  # First quartile (25th percentile)
Q3 = df['Wasserverbrauch'].quantile(0.75)  # Third quartile (75th percentile)
IQR = Q3 - Q1  # Interquartile Range

# Define outlier thresholds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = (df['Wasserverbrauch'] < lower_bound) | (df['Wasserverbrauch'] > upper_bound)

# Interpolate outliers
df['Wasserverbrauch'] = df['Wasserverbrauch'].where(~outliers).interpolate(method='linear')

# Convert all columns to numeric (float64), invalid values will be set as NaN
df = df.apply(pd.to_numeric, errors='coerce')

# Create dummy variables for weekdays
# df['is_monday'] = (df.index.dayofweek == 0).astype(int)
# df['is_tuesday'] = (df.index.dayofweek == 1).astype(int)
# df['is_wednesday'] = (df.index.dayofweek == 2).astype(int)
# df['is_thursday'] = (df.index.dayofweek == 3).astype(int)
# df['is_friday'] = (df.index.dayofweek == 4).astype(int)
df['is_saturday'] = (df.index.dayofweek == 5).astype(int)
df['is_sunday'] = (df.index.dayofweek == 6).astype(int)


# Display the column names
print(df.columns)
print(df.head)
# Check the frequency of the index
print(df.index.freq)
