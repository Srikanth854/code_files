import pandas as pd
import numpy as np

# Load the dataset
data_path = 'final dataset.csv'
data = pd.read_csv(data_path)

# Replace -200 with NaN
data.replace(-200, np.nan, inplace=True)

# Drop the 'NMHC(GT)' column
data.drop('NMHC(GT)', axis=1, inplace=True)

# Fill missing values with median
for column in data.columns:
    if data[column].dtype == float or data[column].dtype == int:
        median_value = data[column].median()
        data[column].fillna(median_value, inplace=True)

# Convert 'Date & Time' to datetime format
data['Date & Time'] = pd.to_datetime(data['Date & Time'])

# Save the cleaned dataset
cleaned_data_path = 'cleaned_dataset.csv'
data.to_csv(cleaned_data_path, index=False)

print("Data cleaning completed and file saved to:", cleaned_data_path)
