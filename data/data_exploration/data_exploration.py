# data_exploration.py

import pandas as pd
from data_exploration.plot_functions import plot_aggregated_data, plot_correlation_matrix, plot_moving_average, plot_acf_pacf

def perform_data_exploration(df):
    """
    Performs data exploration tasks and plots relevant visualizations.
    """
    # Plot aggregated data (7-day sum by default)
    plot_aggregated_data(df, days=7, column='Wasserverbrauch')
    
    # Show moving average with a 30-day window
    plot_moving_average(df, column='Wasserverbrauch', window=30)
    
    # Check correlation matrix and visualize it
    plot_correlation_matrix(df, annot=True, cmap='coolwarm', fmt='.2f')
    
    # Plot ACF and PACF to check for seasonality
    plot_acf_pacf(df, column='Wasserverbrauch', lags=100)
