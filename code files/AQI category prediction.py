import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

def print_colored_category(category):
    """
    Prints the category in a color depending on its value.
    """
    if category == "Good":
        # Green
        print(f"\033[92mPredicted AQI Category : {category}\033[0m")
    elif category == "Moderate":
        # Yellow
        print(f"\033[93mPredicted AQI Category : {category}\033[0m")
    elif category == "Unhealthy for sensitive groups":
        # Red
        print(f"\033[91mPredicted AQI Category : {category}\033[0m")
    else:
        # Default to normal
        print(f"Predicted AQI Category : {category}")

# Load the dataset
data_path = 'updated_dataset_with_AQI.csv'
data = pd.read_csv(data_path)

# Convert 'Date & Time' to datetime and extract year, month, day, and hour
data['Date & Time'] = pd.to_datetime(data['Date & Time'])
data['Year'] = data['Date & Time'].dt.year
data['Month'] = data['Date & Time'].dt.month
data['Day'] = data['Date & Time'].dt.day
data['Hour'] = data['Date & Time'].dt.hour
data.drop('Date & Time', axis=1, inplace=True)  # Drop the original 'Date & Time' column

# Define the specific columns to use
specific_columns = [
    'CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 
    'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'wspd'
]

# Prepare the data
X = data[specific_columns]  # Use only specific columns as features
y = data['AQI_Category']  # Target

# Encode categorical target variable
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Splitting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creating and training the model
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_scaled, y_train)

# Feature Importance
feature_importances = pd.Series(classifier.feature_importances_, index=X.columns)
print("Feature Importances:\n", feature_importances.sort_values(ascending=False))

# User input prompt and prediction
user_input = []
for feature in specific_columns:
    value = float(input(f"Enter the value for {feature}: "))
    user_input.append(value)

# Convert user input into a numpy array and reshape for a single sample
user_input_array = np.array(user_input).reshape(1, -1)

# Standardize the user input using the pre-fitted scaler
user_input_scaled = scaler.transform(user_input_array)

# Predict the AQI category
predicted_category_index = classifier.predict(user_input_scaled)
predicted_category = encoder.inverse_transform(predicted_category_index)[0]
print_colored_category(predicted_category)
print("Predicted AQI Category:", predicted_category)

