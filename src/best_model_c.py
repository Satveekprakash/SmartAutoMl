def best_models(results):
    best_model = None
    best_test_acc = 0

    for model, scores in results.items():
        train_acc = scores["train"]
        test_acc = scores["test"]

        # check overfitting (difference should be small)
        if abs(train_acc - test_acc) < 0.1:
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model = model

    return best_model, best_test_acc
