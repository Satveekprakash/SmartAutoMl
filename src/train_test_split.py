from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def train_split(data,target):
    X = data.drop(columns=[target])
    Y = data[target]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train, X_test, Y_train, Y_test ,X_train_scaled, X_test_scaled
