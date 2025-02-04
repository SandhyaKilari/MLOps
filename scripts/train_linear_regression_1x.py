import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import numpy as np

sampregdata = pd.read_csv('data/sampregdata.csv')

correlation = sampregdata.corr()['y'].drop('y')  # Exclude target itself

best_feature = correlation.abs().idxmax()

# Define input (best feature) and target variable
X = sampregdata[[best_feature]]  # Selecting the best single feature as input
y = sampregdata['target']

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # RMSE is the square root of MSE
r2 = r2_score(y_test, y_pred)  # R² Score

# Print model performance
print(f"Model Performance Metrics:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

# Save the model using pickle
with open("models/linear_model_1x.pkl", "wb") as f:
    pickle.dump(model, f)
