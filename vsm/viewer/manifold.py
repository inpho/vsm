from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering
from sklearn.manifold import Isomap, MDS

from plotting import plot_clusters

from labeleddata import IndexedSymmArray


class Manifold(IndexedSymmArray):
    def __init__(self, dismat, labels=None, cls=[], pos=[]):
        self.dismat = dismat
        self.labels = labels
        self._cls = cls     # Clusters info
        self.pos = pos


    @property
    def cls(self):
        """
        views clusters as lists
        """
        return [[self.labels[i] for i,lab in enumerate(self._cls) if lab == x]
                for x in set(self._cls)]


#
# Clustering methods
#
    def KMeans(self, n_clusters=10, init='k-means++', max_iter=100,
               n_init=1, verbose=1):

        if len(self.pos)==0:
            raise Exception('K-Means requires low dimentional coordinates. Try mds() or isomap() first.')

        model = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, 
                       n_init=n_init,verbose=verbose).fit(self.pos)
        self._cls = model.labels_
        return self.cls
        

    def AffinityPropagation(self):
        model = AffinityPropagation(affinity='precomputed').fit(self.dismat)
        self._cls = model.labels_
        return self.cls


    def SpectralClustering(self, n_clusters=10):
        model = SpectralClustering(n_clusters=n_clusters, 
                                   affinity='precomputed').fit(self.dismat)
        self._cls = model.labels_
        return self.cls



#
# Manifold learning methods
#

    def mds(self, n_components=2, dissimilarity='precomputed'): 
        """
        This requires sklearn ver 0.14 due to dissimilarity argument.
        """
        model = MDS(n_components=n_components, dissimilarity=dissimilarity, max_iter=100)
        self.pos = model.fit_transform(self.dismat)

        return self.pos



    def isomap(self, n_components=2, n_neighbors=3):
        model = Isomap(n_components=n_components, n_neighbors=n_neighbors)
        self.pos  = model.fit(self.dismat).embedding_
        return self.pos



    def plot(self):
        """
        Outputs 2d embeded plot based on the first two coordinates of `pos`
        """
        return plot_clusters(self.pos[:,[0,1]], self.labels, clusters=self._cls)
