import pandas as pd

def aggregate_data(input_path, output_path, frequency='ME'):
    # Read the data into a pandas DataFrame
    df = pd.read_csv(input_path, delimiter=";")

    # Convert 'Datum' column to datetime with the correct format
    df['Datum'] = pd.to_datetime(df['Datum'], format='%d.%m.%Y')

    # Set 'Datum' as the index of the DataFrame (before resampling)
    df.set_index('Datum', inplace=True)

    # Convert all columns except 'Datum' to numeric
    df = df.apply(pd.to_numeric, errors='coerce', axis=1)

    # Perform the resampling and aggregation
    df_resampled = df.resample(frequency).agg({
        'Wasserverbrauch': 'sum',
        'Wegzüge': 'sum',
        'Zuzüge': 'sum',
        'Geburte': 'sum',
        'Todesfälle': 'sum',
        'RainDur_min': 'sum',
        'StrGlo_W/m2': 'sum',
        'T_C': 'mean',
        'T_max_h1_C': 'mean',
        'p_hPa': 'mean'
    })

    # Round the mean columns to 2 decimal places
    df_resampled[['T_C', 'T_max_h1_C', 'p_hPa']] = df_resampled[['T_C', 'T_max_h1_C', 'p_hPa']].round(2)

    # Save the aggregated data to the new CSV file with the ';' delimiter
    df_resampled.to_csv(output_path, sep=";", index=True)

# Example usage for monthly aggregation
input_path = r'C:\Users\emreo\Desktop\ZHAW\DS\water_consumption_2015_2023.csv'
output_path_monthly = r'C:\Users\emreo\Desktop\ZHAW\DS\water_consumption_2015_2023_monthly.csv'
aggregate_data(input_path, output_path_monthly, frequency='ME')  # For monthly aggregation

# Example usage for weekly aggregation
output_path_weekly = r'C:\Users\emreo\Desktop\ZHAW\DS\water_consumption_2015_2023_weekly.csv'
aggregate_data(input_path, output_path_weekly, frequency='W')  # For weekly aggregation

# Example usage for weekly aggregation
output_path_weekly = r'C:\Users\emreo\Desktop\ZHAW\DS\water_consumption_2015_2023_daily.csv'
aggregate_data(input_path, output_path_weekly, frequency='D')  # For daily aggregation
