import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import numpy as np

sampregdata = pd.read_csv('data/sampregdata.csv')

correlation = sampregdata.corr()['y'].drop('y')

best_feature = correlation.abs().idxmax()

X = sampregdata[[best_feature]]  
y = sampregdata['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse) 
r2 = r2_score(y_test, y_pred) 

# Print model performance
print(f"Model Performance Metrics:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R-squared (R²): {r2:.4f}")

with open("models/linear_model_1x.pkl", "wb") as f:
    pickle.dump(model, f)
