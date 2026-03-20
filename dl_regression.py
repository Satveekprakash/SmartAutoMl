def run_regression_dl(data,target):
    #from src.data_loader import load_data
    from src.preprocessing_r import preprocessing
    from src.train_test_split_r import train_split

    import tensorflow as tf
    tf.random.set_seed(3)
    from tensorflow import keras

    from pathlib import Path
    import numpy as np

    # BASE_DIR = Path(__file__).resolve().parent
    # data_path = BASE_DIR / "data" / "sample.csv"
    #
    # # #load
    # data = load_data(data_path)
    # target = "target"

    # #preprocessing
    data = preprocessing(data,target)

    # #split
    X_train, X_test, Y_train, Y_test, X_train_scaled, X_test_scaled = train_split(data, target)

    # #parameter tuning
    neurons_list = [64, 128, 256, 512]
    lr_list = [0.001, 0.0005]

    best_loss = float("inf")
    best_config = None
    best_model = None
    best_mae=None

    # #loop
    for neurons in neurons_list:
        for lr in lr_list:

            print(f"Training: neurons={neurons}, lr={lr}")

            input_dim = X_train_scaled.shape[1]

            # #model
            model = keras.Sequential([

                # #input layer
                keras.Input(shape=(input_dim,)),

                # #hidden layer
                keras.layers.Dense(neurons, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.3),

                # #hidden layer
                keras.layers.Dense(neurons//2, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(0.3),

                # #hidden layer
                keras.layers.Dense(32, activation="relu"),

                # #output layer
                keras.layers.Dense(1)

            ])

            # #compile
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=lr),
                loss="mse",
                metrics=["mae"]
            )

            # #early stopping
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )

            # #train
            model.fit(
                X_train_scaled,
                Y_train,
                validation_data=(X_test_scaled, Y_test),
                epochs=40,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0
            )

            # #evaluate
            loss, mae = model.evaluate(X_test_scaled, Y_test, verbose=0)

            print(f"MAE: {mae}")

            # #best model
            if loss < best_loss:
                best_loss = loss
                best_config = {
                    "neurons": neurons,
                    "learning_rate": lr
                }
                best_model = model
                best_mae = mae


                # #final evaluation
    loss, mae = best_model.evaluate(X_test_scaled, Y_test, verbose=0)
    print("Best MAE:", mae)

    # #simple accuracy
    pred = best_model.predict(X_test_scaled)

    threshold = 5   # #range
    accuracy = np.mean(np.abs(pred.flatten() - Y_test) < threshold)

    print("Accuracy (+/- {}):".format(threshold), accuracy)

    # #save model
    best_model.save("best_model_regression.h5")
    return {
        "model": best_model,
        "best_config": best_config,
        "mse_loss": best_loss,
        "mae": best_mae,
        "accuracy": accuracy,
        "threshold": threshold,
        "features": list(X_train.columns)
    }

