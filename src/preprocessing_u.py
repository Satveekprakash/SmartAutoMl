from sklearn.preprocessing import LabelEncoder

def preprocessing(data):
    data = data.copy()

    missing_ratio = data.isnull().sum().sum() / data.size

    # Handle missing values
    if missing_ratio < 0.05:
        data = data.dropna()
    else:
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = data[col].fillna(data[col].mode()[0])
            else:
                data[col] = data[col].fillna(data[col].mean())

    # Encode categorical variables
    for col in data.select_dtypes(include=['object']):
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    return data