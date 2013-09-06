from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering
from sklearn.manifold import Isomap


def clustering(mat, n_clusters, method="Kmeans"):
    """
    Spectral clustring works with a similarity matrix.
    NB: computing K-means out of similarity matrix is not mathematically sound.
    """
    if method == 'affinity':
        af = AffinityPropagation(affinity='precomputed').fit(mat)
        labels = af.labels_
    elif method == 'spectral':
        sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        sc = sc.fit(mat)
        labels = sc.labels_
    else:
        km = KMeans(n_clusters=n_clusters, init='k-means++', 
                    max_iter=100, n_init=1,verbose=1)
        km.fit(mat)
        labels = list(km.labels_)
    
    return labels


def isomap(mat, n_components=2, n_neighbors=3):
    imap = Isomap(n_components=n_components, n_neighbors=n_neighbors)
    pos  = imap.fit(mat).embedding_
    return pos



