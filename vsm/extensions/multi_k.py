import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
from random import randrange

# generate sample data
n_samples = 30
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
blobs =  datasets.make_blobs(n_samples=n_samples, random_state=8)

# S = noisy_circles[0]
S = blobs[0]


def multik_proto(S, N=10, D_max=30):
    """
    S : set S of r points in Rn.
    N : number N of clusterings to perform.
    D : numbers of clusters.
        <- this should be a distribution over 0,...,N. 
        Let us use a uniform integer for a moment. 
    C : weight cutoff, 0 <= C <= 1.
    
    could make N,D,C optional by setting default values.
    
    Output : 1. A category function 
                C : S -> Z
            2. A cut plot
                f : [0,1] -> Z
    """
    r = len(S)
    M = np.zeros((r, r))
    
    for n in xrange(N):
        # select integer d from (1, 2,..., D_max) 
        d = randrange(D_max) + 1
        km = KMeans(n_clusters=d, init='k-means++',
                    max_iter=100, n_init=1,verbose=1)
        km.fit(S)
        labels = km.labels_

        for i in range(r):
            for j in range(i,r):
                if labels[i] == labels[j]:
                    M[i][j] += 1
                    M[j][i] += 1
                    
    # normalize M[i][j] 
    M /= N

    return M



def cutplot(S, M, N=10):
    from scipy.sparse import csgraph as cs

    # build Weight matrix, W.
    W = M
    cutplot = []
    for l in xrange(N):
        # Construct graph G for which M[i][j] > l/N
        G = M > l * 1.0 /N
        
        n_comp, label = cs.cs_graph_components(G)
        cutplot.append((l * 1.0 /N, n_comp))
    return cutplot 
