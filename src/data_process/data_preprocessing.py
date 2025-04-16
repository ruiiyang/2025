import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load and return the dataset."""
    return pd.read_csv(file_path)

def preprocess_data(data, target_cols=['Y', 'Ya', 'Yb', 'Yc'], random_state=5117):
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

    # Split data
    X = data.drop(columns=target_cols)
    y = data['Y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'label_encoders': label_encoders,
        'scaler': scaler
    }


