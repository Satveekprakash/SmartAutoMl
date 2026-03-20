from sklearn.preprocessing import LabelEncoder

def preprocessing(data, target_col):
    data = data.copy()

    missing_ratio = data.isnull().sum().sum() / data.size

    # Handle missing values
    if missing_ratio < 0.05:
        data = data.dropna()
    else:
        for col in data.columns:
            if col == target_col:
                continue

            if data[col].dtype == 'object':
                data[col] = data[col].fillna(data[col].mode()[0])
            else:
                data[col] = data[col].fillna(data[col].mean())

    # Encode categorical features (NOT target)
    for col in data.select_dtypes(include=['object']):
        if col == target_col:
            continue

        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    return data