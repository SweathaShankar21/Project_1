# Project_1
Predictive Healthcare Analytics: Developed a system that predicts the likelihood of certain medical conditions based on patient data, helping healthcare professionals make informed decisions.
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset 
data = {
    'Age': [45, 50, 60, 35, 28, 48, 52, 42, 33, 29],
    'BMI': [25.5, 30.0, 27.8, 22.3, 26.7, 29.1, 31.2, 23.0, 24.8, 28.4],
    'Smoker': [1, 0, 1, 0, 1, 0, 0, 1, 0, 1],  # 1 for smokers, 0 for non-smokers
    'MedicalCondition': [1, 0, 1, 0, 1, 0, 0, 1, 0, 1],  # 1 if condition present, 0 if not
}

# Create a DataFrame from the sample data
df = pd.DataFrame(data)

# Split the data into features (X) and the target variable (y)
X = df[['Age', 'BMI', 'Smoker']]
y = df['MedicalCondition']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print the results
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)
