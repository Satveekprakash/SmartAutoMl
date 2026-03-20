from models.decision_tree import decision_tree
from models.random_forest import random_forest
from models.k_nearest_neighbors import Knn
from models.svm import svm
from models.logistic_regression import logistic_regression
from sklearn.metrics import accuracy_score

def model_selector(X_train, X_test, Y_train, Y_test, X_train_scaled, X_test_scaled):

    # models
    lr = logistic_regression()
    knn = Knn()
    svm_model = svm()
    rf = random_forest()
    dtree = decision_tree()

    # train
    lr.fit(X_train_scaled, Y_train)
    knn.fit(X_train_scaled, Y_train)
    svm_model.fit(X_train_scaled, Y_train)
    rf.fit(X_train, Y_train)
    dtree.fit(X_train, Y_train)

    # test prediction
    lr_test = lr.predict(X_test_scaled)
    knn_test = knn.predict(X_test_scaled)
    svm_test = svm_model.predict(X_test_scaled)
    rf_test = rf.predict(X_test)
    dt_test = dtree.predict(X_test)

    # train prediction
    lr_train = lr.predict(X_train_scaled)
    knn_train = knn.predict(X_train_scaled)
    svm_train = svm_model.predict(X_train_scaled)
    rf_train = rf.predict(X_train)
    dt_train = dtree.predict(X_train)

    # results
    results = {
        "Logistic Regression": {
            "train": accuracy_score(Y_train, lr_train),
            "test": accuracy_score(Y_test, lr_test)
        },
        "KNN": {
            "train": accuracy_score(Y_train, knn_train),
            "test": accuracy_score(Y_test, knn_test)
        },
        "SVM": {
            "train": accuracy_score(Y_train, svm_train),
            "test": accuracy_score(Y_test, svm_test)
        },
        "Decision Tree": {
            "train": accuracy_score(Y_train, dt_train),
            "test": accuracy_score(Y_test, dt_test)
        },
        "Random Forest": {
            "train": accuracy_score(Y_train, rf_train),
            "test": accuracy_score(Y_test, rf_test)
        }
    }
    #mapping model for user input things
    model_map = {
        "Logistic Regression": lr,
        "KNN": knn,
        "SVM": svm_model,
        "Decision Tree": dtree,
        "Random Forest": rf
    }

    #  BEST MODEL
    best_model_name = max(results, key=lambda x: results[x]["test"])
    best_test_acc = results[best_model_name]["test"]
    best_model = model_map[best_model_name]

    return best_test_acc, best_model_name,best_model, results