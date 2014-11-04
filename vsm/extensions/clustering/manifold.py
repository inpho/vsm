import numpy as np
from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering
from sklearn.manifold import Isomap, MDS
from plotting import plot_clusters


__all__ = [ 'Manifold' ]


class Manifold(object):
    def __init__(self, dismat, labels=None, cls=[], pos=[]):
        self.dismat = np.asarray(dismat)
        self.labels = labels
        self._cls = cls     # Clusters info
        self.pos = pos


    def __str__(self):
        return self.dismat.__str__()


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
               n_init=1, verbose=1, show=True):
        """
        Clusters the objects in `dismat` using k-means algorithm. This requires 
        `pos` be  precomputed by `mds` or `isomap`. For parameters of the 
        algorithms see: 
        http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.
        html#sklearn.cluster.KMeans

        :param n_clusters: Number of clusters used as the parameter for K-means.
        :type n_clusters: int, optional

        :param show: Shows the resulting clusters if true.
        :type n_clusters: boolean, optional
        """

        if len(self.pos)==0:
            raise Exception('K-Means requires low dimentional coordinates. Try mds() or isomap() first.')

        model = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, 
                       n_init=n_init,verbose=verbose).fit(self.pos)
        self._cls = model.labels_

        if show:
            return self.cls
        


    def AffinityPropagation(self, show=True):
        """
        Clusters objects in `dismat` using affinity propagation algorithm.

        :param show: Shows the resulting clusters if true.
        :type n_clusters: boolean, optional
        """

        model = AffinityPropagation(affinity='precomputed').fit(self.dismat)
        self._cls = model.labels_
        
        if show:
            return self.cls



    def SpectralClustering(self, n_clusters=10, show=True):
        """
        Clusters objects in `dismat` using spectral clustering. 

        :param n_clusters: Number of clusters used as the parameter for K-means.
        :type n_clusters: int, optional

        :param show: Shows the resulting clusters if true.
        :type n_clusters: boolean, optional
        """

        model = SpectralClustering(n_clusters=n_clusters, 
                                   affinity='precomputed').fit(self.dismat)
        self._cls = model.labels_

        if show:
            return self.cls



#
# Manifold learning methods
#

    def mds(self, n_components=2, dissimilarity='precomputed', show=False): 
        """
        Calculates lower dimention coordinates using the mds algorithm.
        This requires sklearn ver 0.14 due to the dissimilarity argument.

        :param n_components: dimentionality of the reduced space.
        :type n_components: int, optional

        :param show: Shows the calculated coordinates if true.
        :type show: boolean, optional
        """
        model = MDS(n_components=n_components, dissimilarity=dissimilarity, max_iter=100)
        self.pos = model.fit_transform(self.dismat)

        if show:
            return self.pos



    def isomap(self, n_components=2, n_neighbors=3, show=False):
        """
        Calculates lower dimention coordinates using the isomap algorithm.

        :param n_components: dimentionality of the reduced space
        :type n_components: int, optional

        :param n_neighbors: Used by isomap to determine the number of neighbors
            for each point. Large neighbor size tends to produce a denser map.
        :type n_neighbors: int, optional

        :param show: Shows the calculated coordinates if true.
        :type show: boolean, optional
        """

        model = Isomap(n_components=n_components, n_neighbors=n_neighbors)
        self.pos  = model.fit(self.dismat).embedding_

        if show:
            return self.pos



    def plot(self, xy = (0,1)):
        """
        Outputs 2d embeded plot based on `pos`

        :param xy: specifies the dimsntions of pos to be plotted.
        :type xy: tuple, optional

        """
        return plot_clusters(self.pos[:,[xy[0],xy[1]]], self.labels, clusters=self._cls)
