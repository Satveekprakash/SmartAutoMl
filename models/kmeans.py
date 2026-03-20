from sklearn.cluster import KMeans

def kmeans(k):
    return KMeans(n_clusters=k, random_state=0)