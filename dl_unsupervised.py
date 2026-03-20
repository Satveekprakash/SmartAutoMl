def run_unsupervised(data):

    # from src.data_loader import load_data
    from src.preprocessing_c import preprocessing
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from scipy import stats
    from pathlib import Path

    import tensorflow as tf
    from tensorflow import keras
    import numpy as np

    tf.random.set_seed(3)


    #data
    # BASE_DIR = Path(__file__).resolve().parent
    # data_path = BASE_DIR / "data" / "sample2.csv"
    # data = load_data(data_path)


    # preprocessing
    data = preprocessing(data)

    # clean data
    data = data.drop(columns="id", errors="ignore")
    data = data.loc[:, data.nunique() > 1]

    # standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    # PCA
    pca = PCA(n_components=2)
    X_scaled = pca.fit_transform(X_scaled)

    # remove outliers
    z = np.abs(stats.zscore(X_scaled))
    mask = (z < 3).all(axis=1)
    X_scaled = X_scaled[mask]

    # hyperparameters
    neurons_list = [16, 32, 64]
    lr_list = [0.001, 0.0005]

    best_loss = float("inf")
    best_config = None
    best_model = None

    # training loop
    for neurons in neurons_list:
        for lr in lr_list:

            input_dim = X_scaled.shape[1]

            model = keras.Sequential([
                keras.Input(shape=(input_dim,)),
                keras.layers.Dense(neurons, activation="relu"),
                keras.layers.Dense(neurons // 2, activation="relu"),
                keras.layers.Dense(neurons, activation="relu"),
                keras.layers.Dense(input_dim)
            ])

            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=lr),
                loss="mse"
            )

            early_stop = keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=5,
                restore_best_weights=True
            )

            model.fit(
                X_scaled,
                X_scaled,
                epochs=40,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0
            )

            loss = model.evaluate(X_scaled, X_scaled, verbose=0)

            if loss < best_loss:
                best_loss = loss
                best_config = {
                    "neurons": neurons,
                    "learning_rate": lr,
                    "input_dim": input_dim,
                    "batch_size": 32,
                    "epochs": 40
                }
                best_model = model

    # reconstruction
    reconstructed = best_model.predict(X_scaled)
    error = np.mean((X_scaled - reconstructed) ** 2)

    # save model
    import joblib
    joblib.dump({
        "scaler": scaler,
        "pca":pca,
        "threshold":error,
    },"dl_preprocessing.joblib")
    best_model.save("dl_unsupervised.h5")

    return {
        "model": best_model,
        "scaler": scaler,
        "pca": pca,
        "best_config": best_config,
        "best_loss": best_loss,
        "reconstruction_error": error,
        "features_used": list(data.columns),
        "outlier_mask": mask
    }