from sklearn.preprocessing import LabelEncoder

def preprocessing(data,target):
    missing_ratio=data.isnull().sum().sum()/data.size
    #  data
    data=data.copy()

    if missing_ratio <0.05 :
          data=data.dropna()
    else:
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = data[col].fillna(data[col].mode()[0])
            else:
                data[col] = data[col].fillna(data[col].mean())
    #enode target used for prediction
    target_encoder = None

    if data[target].dtype == 'object':
        target_encoder = LabelEncoder()
        data[target] = target_encoder.fit_transform(data[target])

    # convert text to number
    for col in data.select_dtypes(include=['object']):
        le=LabelEncoder()
        data[col]=le.fit_transform(data[col])

    return data,target_encoder



