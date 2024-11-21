import pandas as pd

def merge_files(file1_path, file2_path, output_path):
    # Read the two CSV files into pandas DataFrames
    df1 = pd.read_csv(file1_path, delimiter=";")
    df2 = pd.read_csv(file2_path, delimiter=";")
    
    # Convert 'Datum' column to datetime format using ISO8601 format
    df1['Datum'] = pd.to_datetime(df1['Datum'], format='%Y-%m-%d')
    df2['Datum'] = pd.to_datetime(df2['Datum'], format='%Y-%m-%d')

    # Merge the two DataFrames on 'Datum' column
    merged_df = pd.merge(df1, df2, on='Datum', how='outer')  # You can use 'inner', 'left', or 'right' for different merge types

    # Save the merged data to a new CSV file
    merged_df.to_csv(output_path, sep=";", index=False)

# Example usage:
file1_path = r'C:\Users\emreo\Desktop\ZHAW\DS\water_consumption_2015_2023_monthly.csv'
file2_path = r'C:\Users\emreo\Desktop\ZHAW\DS\aggregated_bevölkerung.csv'
output_path = r'C:\Users\emreo\Desktop\ZHAW\DS\monthly_with_population.csv'
merge_files(file1_path, file2_path, output_path)
