def run_regression_ml(data,target):
    #from src.data_loader import load_data
    from src.preprocessing_r import preprocessing
    from src.train_test_split_r import train_split
    from src.model_selector_r import model_selector
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from pathlib import Path

    # BASE_DIR = Path(__file__).resolve().parent
    # data_path = BASE_DIR / "data" / "sample3.csv"
    #
    # #load
    # data = load_data(data_path)
    # target ="target"


    # #preprocessing
    data = preprocessing(data,target)


    # #split
    X_train, X_test, Y_train, Y_test, X_train_scaled, X_test_scaled=train_split(data,target)

    # #model selection
    best_model_name,best_model, best_score, results=model_selector(X_train, X_test, Y_train, Y_test ,X_train_scaled, X_test_scaled)

    # pipeling for ui
    scaling_models = ["KNN", "SVR", "Linear Regression"]

    if best_model_name in scaling_models:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", best_model)
        ])
    else:
        # no need of  scaling
        pipeline = Pipeline([
            ("model", best_model)
        ])

    # Fit pipeline
    pipeline.fit(X_train, Y_train)
    import joblib
    joblib.dump( pipeline, 'best_pipeline_ml_r.joblib')

    # #best model
    return  best_model_name, best_score, results,pipeline

