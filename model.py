import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load the dataset
data = pd.read_csv('heart.csv')

# Selecting features and target
X = data[['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol']]
y = data['HeartDisease']

# Convert categorical features to numerical if necessary
X['ChestPainType'] = X['ChestPainType'].map({'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3})

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'model.pkl')
