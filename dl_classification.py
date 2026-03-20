def run_classification_dl(data,target):
    #from src.data_loader import load_data
    from src.preprocessing_c import preprocessing
    from src.train_test_split import train_split

    import tensorflow as tf
    tf.random.set_seed(3)
    from tensorflow import keras

    from pathlib import Path

    # BASE_DIR = Path(__file__).resolve().parent
    # data_path = BASE_DIR / "data" / "sample.csv"
    #
    # # #load
    # data = load_data(data_path)
    # target = "target"

    # #preprocessing
    data = preprocessing(data)

    # #split
    X_train, X_test, Y_train, Y_test, X_train_scaled, X_test_scaled = train_split(data, target)

    # #parameter tuning (fast)
    neurons_list = [64, 128, 256, 512]   # #optional added 512
    lr_list = [0.001, 0.0005]

    best_acc = 0
    best_config = None
    best_model = None

    # #loop
    for neurons in neurons_list:
        for lr in lr_list:

            print(f"Training: neurons={neurons}, lr={lr}")

            input_dim = X_train_scaled.shape[1]
            num_classes = len(set(Y_train))

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
                keras.layers.Dense(num_classes, activation="softmax")

            ])

            # #compile
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=lr),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"]
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
            loss, acc = model.evaluate(X_test_scaled, Y_test, verbose=0)

            print(f"Accuracy: {acc}")

            # #best model
            if acc > best_acc:
                best_acc = acc
                best_config = (neurons, lr)
                best_model = model


    # #result
    return best_model, best_config, best_acc


