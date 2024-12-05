import streamlit as st
import os
import sys

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, "..")))

# Import data loading and exploration functions
from data_exploration.data_loader import load_and_clean_data
from data_exploration.plot_functions import plot_aggregated_data

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the absolute path to the CSV
input_file_path = os.path.join(script_dir, "water_data", "input", "water_consumption_2015_2023.csv")

# Load and clean the data using the data loader
df_cleaned = load_and_clean_data(input_file_path)

# Streamlit tabs
tab1, tab2, tab3 = st.tabs(["Manual Exploration", "Python Visualization", "Prediction"])

with tab1:
    st.header("Manual Data Exploration")

with tab2:
    st.header("Python Visualization")

with tab3:
    st.header("Prediction")
