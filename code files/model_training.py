import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

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

# Prepare the data
X = data.drop(['AQI_Category'], axis=1)  # Use all other columns as features including 'AQI'
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

# Predicting the test set results
y_pred = classifier.predict(X_test_scaled)

# Evaluating the model
print(classification_report(y_test, y_pred, target_names=encoder.classes_))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Cross-Validation to check the model's robustness
cv_scores = cross_val_score(classifier, X_train_scaled, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", cv_scores.mean())

# Feature Importance
feature_importances = pd.Series(classifier.feature_importances_, index=X.columns)
print("Feature Importances:\n", feature_importances.sort_values(ascending=False))

