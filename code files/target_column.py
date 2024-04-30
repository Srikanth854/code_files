import pandas as pd
import numpy as np

# Load the cleaned dataset
data_path = 'cleaned_dataset.csv'
data = pd.read_csv(data_path)

# Define breakpoints for CO in ppm (example values, adjust based on actual EPA standards)
co_breakpoints_ppm = [(0, 4.4), (4.5, 9.4), (9.5, 12.4), (12.5, 15.4), (15.5, 30.4), (30.5, 40.4)]
co_aqi_values = [50, 100, 150, 200, 300, 400]

# Define breakpoints for NO2 in ppb (example values, adjust based on actual EPA standards)
no2_breakpoints_ppb = [(0, 53), (54, 100), (101, 360), (361, 649), (650, 1249)]
no2_aqi_values = [50, 100, 150, 200, 300]

# Function to calculate the sub-index for a pollutant
def calculate_sub_index(concentration, breakpoints, aqi_values):
    for i, (low, high) in enumerate(breakpoints):
        if low <= concentration <= high:
            return aqi_values[i]
    return 0  # return 0 if concentration is below the lowest breakpoint

# Calculate sub-index for each pollutant
data['CO_Sub_Index'] = data['CO(GT)'].apply(calculate_sub_index, args=(co_breakpoints_ppm, co_aqi_values))
data['NO2_Sub_Index'] = data['NO2(GT)'].apply(calculate_sub_index, args=(no2_breakpoints_ppb, no2_aqi_values))

# Calculate the overall AQI
data['AQI'] = data[['CO_Sub_Index', 'NO2_Sub_Index']].max(axis=1)

# Categorize AQI into levels
def categorize_aqi(aqi):
    if aqi <= 50:
        return 'Good'
    elif aqi <= 100:
        return 'Moderate'
    elif aqi <= 150:
        return 'Unhealthy for sensitive groups'
    elif aqi <= 200:
        return 'Unhealthy'
    elif aqi <= 300:
        return 'Very unhealthy'
    else:
        return 'Hazardous'

data['AQI_Category'] = data['AQI'].apply(categorize_aqi)

# Save the updated dataset
cleaned_data_path = 'updated_dataset_with_AQI.csv'
data.to_csv(cleaned_data_path, index=False)

print("Data with AQI levels saved to:", cleaned_data_path)
