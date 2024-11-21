import pandas as pd
import matplotlib.pyplot as plt

# Function to aggregate data
def aggregate_data(df, days=7, column='Wasserverbrauch'):
    """
    Aggregates the specified column in the DataFrame by resampling it to a specified number of days.
    
    Parameters:
    - df: DataFrame containing the data
    - days: Number of days to aggregate (default is 7)
    - column: The column name to aggregate (default is 'Wasserverbrauch')
    
    Returns:
    - Aggregated data (pandas Series)
    """
    # Ensure index is datetime (in case it's not already set)
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index, dayfirst=True)
    
    # If days = 1, return original data without aggregation
    if days == 1:
        return df[column]
    else:
        # Resample data by summing up over the specified number of days
        return df[column].resample(f'{days}D').sum()


# Function to plot aggregated data
def plot_aggregated_data(df, days=7, column='Wasserverbrauch'):
    """
    Plots the aggregated data (sum of a given column over the specified number of days).
    
    Parameters:
    - df: DataFrame containing the data
    - days: Number of days to aggregate (default is 7)
    - column: The column name to plot (default is 'Wasserverbrauch')
    """
    # Get the aggregated data by calling the aggregate_data function
    aggregated_data = aggregate_data(df, days, column)
    
    # Plot the aggregated data
    plt.figure(figsize=(10, 6))
    plt.plot(aggregated_data.index, aggregated_data.values, marker='o', label=f'{days}-Day Aggregated {column}' if days > 1 else f'{column} (Original Data)')
    plt.title(f'{column} Aggregated Over {days}-Day Periods' if days > 1 else f'{column} (Original Data)')
    plt.xlabel('Date')
    plt.ylabel(f'{column} (Aggregated)' if days > 1 else f'{column}')
    plt.legend()
    plt.grid(True)
    plt.show()
