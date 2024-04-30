import pandas as pd
import numpy as np
from datetime import datetime
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
        print(f"\033[92mPredicted AQI Category for {input_date}: {category}\033[0m")
    elif category == "Moderate":
        # Yellow
        print(f"\033[93mPredicted AQI Category for {input_date}: {category}\033[0m")
    elif category == "Unhealthy for sensitive groups":
        # Red
        print(f"\033[91mPredicted AQI Category for {input_date}: {category}\033[0m")
    else:
        # Default to normal
        print(f"Predicted AQI Category for {input_date}: {category}")

# Load the dataset
data_path = 'updated_dataset_with_AQI.csv'
data = pd.read_csv(data_path)

# Convert 'Date & Time' to datetime
data['Date & Time'] = pd.to_datetime(data['Date & Time'])
data['Year'] = data['Date & Time'].dt.year
data['Month'] = data['Date & Time'].dt.month
data['Day'] = data['Date & Time'].dt.day
data['Hour'] = data['Date & Time'].dt.hour

# Assuming we create a time trend feature (e.g., years since 2004)
data['Time Trend'] = data['Year'] - 2004

# Prepare the data
X = data[['Year', 'Month', 'Day', 'Hour', 'Time Trend']]  # Using time-related features
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

# Predicting based on user input date
input_date = input("Enter the date (MM/DD/YYYY): ")
input_datetime = datetime.strptime(input_date, "%m/%d/%Y")
input_features = np.array([[input_datetime.year, input_datetime.month, input_datetime.day, input_datetime.hour, input_datetime.year - 2004]])
input_features_scaled = scaler.transform(input_features)

# Predict the AQI category
predicted_category_index = classifier.predict(input_features_scaled)
predicted_category = encoder.inverse_transform(predicted_category_index)[0]

# Call the function to print the category in the appropriate color
print_colored_category(predicted_category)
#print("Predicted AQI Category for", input_date, ":", predicted_category)

# Evaluation and feature importance (optional, for model evaluation purposes)
y_pred = classifier.predict(X_test_scaled)
print(classification_report(y_test, y_pred, target_names=encoder.classes_))
print("Accuracy on test set:", accuracy_score(y_test, y_pred))


