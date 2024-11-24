import pandas as pd
import glob
import os

# Specify the path with wildcard to read all CSV files in the folder
path = r'C:\Users\emreo\Desktop\ZHAW\DS\meteo_data\*.csv'
all_files = glob.glob(path)

# Concatenate all CSV files into one DataFrame
df_list = [pd.read_csv(file) for file in all_files]
df = pd.concat(df_list, ignore_index=True)

# Pivot the DataFrame to create columns based on 'Parameter' and 'Einheit'
df_pivoted = df.pivot_table(index=['Datum', 'Standort', 'Intervall'], 
                            columns=['Parameter', 'Einheit'], 
                            values='Wert').reset_index()

# Flatten the multi-index columns
df_pivoted.columns = [f"{i}_{j}" if j else i for i, j in df_pivoted.columns]

# Convert 'Datum' to desired date format
df_pivoted['Datum'] = pd.to_datetime(df_pivoted['Datum']).dt.strftime('%d.%m.%Y')

# Filter out specific 'Standort' values
excluded_locations = ['Zch_Rosengartenstrasse', 'Zch_Schimmelstrasse']
df_filtered = df_pivoted[~df_pivoted['Standort'].isin(excluded_locations)]

# Drop specific columns
columns_to_drop = ['Intervall', 'Standort']  # Replace with the actual column names you want to drop
df_filtered = df_filtered.drop(columns=columns_to_drop, errors='ignore')

# Save the filtered DataFrame as a new CSV file in the same path
output_path = os.path.join(os.path.dirname(path), 'filtered_meteo_data.csv')
df_filtered.to_csv(output_path, index=False)