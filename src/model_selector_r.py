from models.model_regression import get_regression_models
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def model_selector(X_train, X_test, Y_train, Y_test, X_train_scaled, X_test_scaled):

    models = get_regression_models()
    results = {}

    for name, model in models.items():

        # -------- Train + Predict --------
        if name in ["KNN", "SVR", "Linear Regression"]:
            model.fit(X_train_scaled, Y_train)
            pred_train = model.predict(X_train_scaled)
            pred_test = model.predict(X_test_scaled)
        else:
            model.fit(X_train, Y_train)
            pred_train = model.predict(X_train)
            pred_test = model.predict(X_test)

        # -------- Metrics --------
        results[name] = {
            "train_r2": r2_score(Y_train, pred_train),
            "test_r2": r2_score(Y_test, pred_test),
            "rmse": np.sqrt(mean_squared_error(Y_test, pred_test))
        }

    # -------- Best Model --------
    best_model_name = max(results, key=lambda x: results[x]["test_r2"])
    best_score = results[best_model_name]["test_r2"]
    best_model = models[best_model_name]

    return best_model_name,best_model, best_score, results