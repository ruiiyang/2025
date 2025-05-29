import pandas as pd
from sklearn.preprocessing import LabelEncoder
from difflib import get_close_matches
pd.set_option('display.max_columns', None)

def load_data(file_path):
    """Load and return the dataset."""
    return pd.read_csv(file_path)


def describe_plus(df: pd.DataFrame, zero_cols: list[str] = None) -> pd.DataFrame:
    desc = df.describe().T[['count', 'mean', 'min', 'max','std']]
    zero_ratios = (df == 0).astype(float).mean()
    # column Y, Ya, Yb, Yc are the column of 0 and 1 value.
    desc['Zero_Ratio (%)'] = desc.index.map(lambda col: round(zero_ratios[col] * 100, 2) if col in zero_ratios else "Not Calculated")
    return desc


def clean_data(data: pd.DataFrame) \
        -> tuple[pd.DataFrame, dict]:

    """
    Preprocess data:
    1. Normalize and correct string values in 'X2'
    2. Encode categorical columns
    3. Drop missing values
    """
    # process X2 correct the value
    # X2 original values are array(['Spoon ', 'Fork ', 'F0rk', 'Soon', 'Sp0on'].
    # - Visible number 0 mistaken as o.
    # - misspelling of spoon as soon
    if 'X2' in data.columns:
        # given correct category
        true_categories = ['spoon', 'fork']

        def normalize(text):
            text = text.strip().lower().replace('0', 'o')
            return ''.join(c for c in text if c.isalpha())

        def correct_value(val):
            cleaned = normalize(str(val))
            match = get_close_matches(cleaned, true_categories, n=1, cutoff=0.6)
            if match:
                return match[0].capitalize()
            else:
                return val

        data['X2'] = data['X2'].apply(correct_value)

    # Encode categorical columns
    label_encoders = {}
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le

    # Drop missing values
    data = data.dropna()

    return data, label_encoders