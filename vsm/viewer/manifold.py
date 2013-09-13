from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering
from sklearn.manifold import Isomap, MDS


def clustering(mat, n_clusters, method="Kmeans"):
    """
    Clusters matrix by a spceificed clustering algorithm. 
    Currently it supports K-means, Spectral Clustering and Affinity
    Propagation algorithms. K-means and spectral clustering cluster
    topics into a given number of clusters, whereas affinity
    propagation does not require the fixed cluster number. 

    NB: computing K-means out of similarity matrix is not mathematically sound.

    Parameters
    ----------

    """
    if method == 'affinity':
        model = AffinityPropagation(affinity='precomputed').fit(mat)
    elif method == 'spectral':
        model = SpectralClustering(n_clusters=n_clusters, 
                                   affinity='precomputed').fit(mat)
    else:
        model = KMeans(n_clusters=n_clusters, init='k-means++', 
                    max_iter=100, n_init=1,verbose=1).fit(mat)

    return list(model.labels_)



def mds(mat, n_components=2, dissimilarity='precomputed'): 
    """
    This requires sklearn ver 0.14 due to dissimilarity argument.
    """
    model = MDS(n_components=n_components, dissimilarity=dissimilarity, max_iter=100)
    pos = model.fit_transform(mat)
    return pos



def isomap(mat, n_components=2, n_neighbors=3):
    model = Isomap(n_components=n_components, n_neighbors=n_neighbors)
    pos  = model.fit(mat).embedding_
    return pos



