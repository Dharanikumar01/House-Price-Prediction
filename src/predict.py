import joblib
import pandas as pd

# Load the trained model
model = joblib.load("../models/house_price_model.pkl")

# Sample house details
sample_house = pd.DataFrame({
    "bedrooms": [3],
    "bathrooms": [2],
    "sqft_living": [1800],
    "sqft_lot": [5000],
    "floors": [1],
    "waterfront": [0],
    "view": [0],
    "condition": [3],
    "sqft_above": [1800],
    "sqft_basement": [0],
    "yr_built": [2005],
    "yr_renovated": [0],
    "city": [10],        # Encoded value
    "statezip": [15]     # Encoded value
})

# Predict price
predicted_price = model.predict(sample_house)

print(f"Predicted House Price: ${predicted_price[0]:,.2f}")