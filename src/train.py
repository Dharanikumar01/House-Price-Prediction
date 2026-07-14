import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load dataset
df = pd.read_csv("../data/Housing.csv")

# Drop unnecessary columns
df = df.drop(columns=["date", "street", "country"])

# Encode categorical columns
le = LabelEncoder()

df["city"] = le.fit_transform(df["city"])
df["statezip"] = le.fit_transform(df["statezip"])

# Split features and target
X = df.drop("price", axis=1)
y = df["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation")
print("----------------")
print(f"MAE : {mae}")
print(f"MSE : {mse}")
print(f"RMSE: {rmse}")
print(f"R² Score: {r2}")

# Save model
joblib.dump(model, "../models/house_price_model.pkl")

print("\nModel saved successfully!")