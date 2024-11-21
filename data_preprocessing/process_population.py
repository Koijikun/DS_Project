import pandas as pd

# File path
path = r'C:\Users\emreo\Desktop\ZHAW\DS\bevölkerung_monatlich.csv'

# Read the data into a pandas DataFrame
df = pd.read_csv(path, delimiter=";")

# Convert the 'Datum' column to datetime format
df['Datum'] = pd.to_datetime(df['Datum'], format='%d.%m.%Y')

# Extract the month-year from the 'Datum' column
df['Monat'] = df['Datum'].dt.to_period('M')

# Group by 'Monat' and 'Herkunft', and aggregate 'Bevölkerung' by summing
aggregated_data = df.groupby(['Monat', 'Herkunft'])['Bevölkerung'].sum().reset_index()

# Convert the 'Monat' (period) to the last day of each month
aggregated_data['Datum'] = aggregated_data['Monat'].dt.to_timestamp('M')

# Drop the 'Monat' column as it's no longer needed
aggregated_data.drop(columns=['Monat'], inplace=True)

# Reorder columns to match the desired order: 'Datum', 'Herkunft', 'Bevölkerung'
aggregated_data = aggregated_data[['Datum', 'Herkunft', 'Bevölkerung']]

# Sort the data by 'Datum' to ensure correct order
aggregated_data.sort_values(by='Datum', ascending=True, inplace=True)

# Define the new path for the output CSV file
output_path = r'C:\Users\emreo\Desktop\ZHAW\DS\aggregated_bevölkerung.csv'

# Save the aggregated data to the new CSV file with ';' delimiter
aggregated_data.to_csv(output_path, index=False, sep=";")
