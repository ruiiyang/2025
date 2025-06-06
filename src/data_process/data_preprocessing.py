from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split


def preprocess_data(
    data,
    label_encoders,
    target_cols=['Y','Ya','Yb','Yc'],
    random_state=5117
):
    """
    Preprocess data using the data from previously cleaned df.
    This step is to better handle Ya Yb Yc effects to Y

    """

    # Remove Y outliers using historical data (IQR method)
    historical_cols = ['Ya', 'Yb', 'Yc']
    q1 = data[historical_cols].quantile(0.25, axis=1)
    q3 = data[historical_cols].quantile(0.75, axis=1)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    valid_mask = (data['Y'] >= lower_bound) & (data['Y'] <= upper_bound)
    data = data[valid_mask]

    # Split data
    X = data.drop(columns=target_cols)
    y = data['Y']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Scale features
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    scaler = RobustScaler()
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

