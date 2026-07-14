import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    # Drop unnecessary columns
    df = df.drop(columns=["date", "street", "country"])

    # Encode categorical columns
    le_city = LabelEncoder()
    le_statezip = LabelEncoder()

    df["city"] = le_city.fit_transform(df["city"])
    df["statezip"] = le_statezip.fit_transform(df["statezip"])

    return df