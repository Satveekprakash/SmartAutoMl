from sklearn.cluster import DBSCAN


def dbscan(eps):
    return DBSCAN(eps=eps, min_samples=5)