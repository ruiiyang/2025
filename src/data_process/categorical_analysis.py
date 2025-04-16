from sklearn.preprocessing import LabelEncoder

def encode_categorical(data):
    """对分类变量进行标签编码"""
    le = LabelEncoder()
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = le.fit_transform(data[col].astype(str))
    return data


