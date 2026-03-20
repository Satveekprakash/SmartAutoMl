from models.dbscan import dbscan
from models.kmeans import kmeans
from models.agglomerative_clustering import agglomerative
from sklearn.metrics import silhouette_score


def model_selector(data):

    # -------- Model 1: KMeans --------
    best_k_value = 2
    best_k_score = -1
    best_k_model = None
    for k in range(2, 11):
        model = kmeans(k)
        labels = model.fit_predict(data)

        score = silhouette_score(data, labels)

        if score > best_k_score:
            best_k_value = k
            best_k_score = score
            best_k_model = model

            # -------- Model 2: DBSCAN --------
    best_db_score = -1
    best_eps = None
    best_db_model = None

    for eps in [0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        model = dbscan(eps)
        labels = model.fit_predict(data)

        # Avoid single cluster case
        if len(set(labels)) > 1:
            score = silhouette_score(data, labels)

            if score > best_db_score:
                best_db_score = score
                best_eps = eps
                best_db_model = model

             # -------- Model 3: Agglomerative --------
    agg_model = agglomerative()
    agg_labels = agg_model.fit_predict(data)

    if len(set(agg_labels)) > 1:
        agg_score = silhouette_score(data, agg_labels)
    else:
        agg_score = -1

    # -------- Compare All --------
    results = {
        "KMeans": best_k_score,
        "DBSCAN": best_db_score,
        "Agglomerative": agg_score
    }
    # map models
    model_map = {
        "KMeans": best_k_model,
        "DBSCAN": best_db_model,
        "Agglomerative": agg_model
    }
    best_model_name = max(results, key=results.get)
    best_model = model_map[best_model_name]
    return best_model_name,best_model, results