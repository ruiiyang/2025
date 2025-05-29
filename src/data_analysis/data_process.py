import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """Load and return the dataset."""
    return pd.read_csv(file_path)

def clean_data(data):
    
    """
    Preprocess data:
    1. Encode categorical columns
    2. Handle missing values
    3. Split into train/test sets
    """
    # Encode categorical columns
    label_encoders = {}
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le

    # Handle missing values
    data = data.dropna()

    return data, label_encoders