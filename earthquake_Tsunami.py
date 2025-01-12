import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
data = pd.read_csv('earthquake_data.csv')
print(data.head())
print(data.info())
print( data.isnull().sum())
data["alert"] = data["alert"].fillna("red")
data.isnull().sum()
del data['title']
del data['country']
del data['continent']
del data['location']
data.head()
data.isnull().sum()
data['date_time'] = pd.to_datetime(data['date_time'])
data['year'] = data['date_time'].dt.year
data['month'] = data['date_time'].dt.month
data['day'] = data['date_time'].dt.day
data['hour'] = data['date_time'].dt.hour
categorical_cols = ['alert', 'magType']
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
data = data.drop(['date_time', 'net'], axis=1)
# Define features (X) and target (y)
X = data.drop('tsunami', axis=1)  # 'tsunami' is the target column
y = data['tsunami']
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
# Step 7: Model Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
# Step 8: Save the Model
joblib.dump(model, 'earthquake_tsunami_predictor.pkl')
# Save model columns
joblib.dump(X_train.columns.tolist(), 'model_columns.pkl')

print("Model saved as 'earthquake_tsunami_predictor.pkl'")