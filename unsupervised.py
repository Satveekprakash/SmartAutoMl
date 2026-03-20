def run_unsupervised(data):
    #from src.data_loader import load_data
    from src.preprocessing_u import preprocessing
    from src.model_selector_u import model_selector
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from pathlib import Path

    # BASE_DIR = Path(__file__).resolve().parent
    # data_path = BASE_DIR / "data" / "sample2.csv"
    #
    # #load
    # data = load_data(data_path)


    # #preprocessing
    data = preprocessing(data)

    data=data.drop(columns="id",errors="ignore")
    data=data.loc[:,data.nunique()>1]

    # #standarization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
     #reduce dimmenson to 2d
    pca = PCA(n_components=2)
    X_scaled = pca.fit_transform(X_scaled)

    #to remove outliers
    from scipy import stats
    import numpy as np
    z=np.abs(stats.zscore(X_scaled))
    X_scaled=X_scaled[(z<3).all(axis=1)]


    # #model
    best_model_name,best_model, results=model_selector(X_scaled)

    #pipeline to show input in ui
    scaling_models = ["KMeans", "DBSCAN", "Agglomerative"]

    if best_model_name in scaling_models:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2)),
            ("model", best_model)
        ])
    else:
       pass

    # Fit pipeline
    pipeline.fit(data)

    #cluster summary so user can understand its impotnace
    labels = pipeline.named_steps["model"].fit_predict(
        pipeline.named_steps["pca"].transform(
            pipeline.named_steps["scaler"].transform(data)
        )
    )

    data_with_cluster = data.copy()
    data_with_cluster["cluster"] = labels

    cluster_summary = data_with_cluster.groupby("cluster").mean()
    import joblib
    joblib.dump(pipeline, 'best_pipeline_ml_u.joblib')

    return best_model_name,best_model,results,pipeline, cluster_summary


