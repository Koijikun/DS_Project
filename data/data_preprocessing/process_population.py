import pandas as pd
import os

# Define the base path
base_path = os.getcwd()

# Define the input and output paths
input_path = os.path.join(base_path, 'water_data', 'input', 'bevölkerung_monatlich.csv')
output_path = os.path.join(base_path, 'water_data', 'output', 'population_monthly.csv')

# Read the data
df = pd.read_csv(input_path, delimiter=";")

# Convert the 'Datum' column to datetime
df['Datum'] = pd.to_datetime(df['Datum'], format='%d.%m.%Y')

# Extract the month from the 'Datum' column
df['Monat'] = df['Datum'].dt.to_period('M')

# Group by 'Monat' and 'Herkunft', and aggregate 'Bevölkerung' by summing
aggregated_data = df.groupby(['Monat', 'Herkunft'])['Bevölkerung'].sum().reset_index()

# Convert the 'Monat' to the last day of each month
aggregated_data['Datum'] = aggregated_data['Monat'].dt.to_timestamp('M')

# Drop the 'Monat' column
aggregated_data.drop(columns=['Monat'], inplace=True)

# Reorder columns
aggregated_data = aggregated_data[['Datum', 'Herkunft', 'Bevölkerung']]

# Sort the data by 'Datum'
aggregated_data.sort_values(by='Datum', ascending=True, inplace=True)

# Save the aggregated data
aggregated_data.to_csv(output_path, index=False, sep=";")
