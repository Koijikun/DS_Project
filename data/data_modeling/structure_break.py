import sys
import os
import pandas as pd

# Add the parent directory to system path for module imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

# Import data loading and exploration functions
from data_exploration.data_loader import load_and_clean_data
from data_exploration.plot_functions import plot_aggregated_data
# Define the base path
base_path = os.getcwd()

# Specify the file path (can be modified to load different files)
input_file_path = os.path.join(base_path, "water_data", "input", "water_consumption_2015_2023.csv")

# Load and clean the data using the data loader
df_cleaned = load_and_clean_data(input_file_path)

# Define the start and end date of the structural break
break_start = "2020-01-01"
break_end = "2020-12-31"

# Identify the rows corresponding to the structural break period
mask_break_period = (df_cleaned.index >= break_start) & (df_cleaned.index <= break_end)

# Adjust the data for the structural break period
# For example: Let's adjust 'Wasserverbrauch' by normalizing it with the mean of the years before and after 2020
mean_before_break = df_cleaned.loc[df_cleaned.index < break_start, 'Wasserverbrauch'].mean()
mean_after_break = df_cleaned.loc[df_cleaned.index > break_end, 'Wasserverbrauch'].mean()
mean_adjustment_factor = (mean_before_break + mean_after_break) / 2

df_cleaned.loc[mask_break_period, 'Wasserverbrauch'] = (
    df_cleaned.loc[mask_break_period, 'Wasserverbrauch']
    / df_cleaned.loc[mask_break_period, 'Wasserverbrauch'].mean()
    * mean_adjustment_factor
)

# Insert the adjusted values back into the DataFrame
df_cleaned.update(df_cleaned.loc[mask_break_period])

# Optional: Verify adjustments
print("Adjusted Data for Structural Break Period:")
print(df_cleaned.loc[mask_break_period])

plot_aggregated_data(df_cleaned)
