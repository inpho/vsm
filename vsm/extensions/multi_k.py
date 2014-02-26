import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
from random import randrange
from scipy.sparse import csgraph as cs


# generate sample data
n_samples = 30
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
blobs =  datasets.make_blobs(n_samples=n_samples, random_state=8)

# S = noisy_circles[0]
S = blobs[0]

def multi_k(S, N=30, D=(10,30), C=None):
    import matplotlib.pyplot as plt

    M, cutplot = connection_matrix(S=S, N=N, D=D)

    if C == None:
        C = find_cutoff(cutplot)
    print "Weight cutoff is set to ", C
    category_func = category_mat(S, M, cutplot, C=C)

    x, y = zip(*cutplot)
    plt.plot(x,y)

    return plt, cutplot, category_func


def connection_matrix(S, N=10, D=(10,30)):
    """
    S : set S of r points in Rn.
    N : number N of clusterings to perform.
    D : numbers of clusters.
        <- this should be a distribution over 0,...,N. 
        Let us use a uniform integer for a moment. 
    C : weight cutoff, 0 <= C <= 1.
    
    could make N,D,C optional by setting default values.
    
    :returns: M : connection matrix 
             cutplot :  A cut plot
                f : [0,1] -> Z
    """
    r = len(S)
    M = np.zeros((r, r))
    
    for n in xrange(N):
        # select integer d from D[0] to D[1] 
        d = randrange(D[0],D[1]+1)
        km = KMeans(n_clusters=d, init='k-means++',
                    max_iter=100, n_init=1,verbose=False)
        km.fit(S)
        labels = km.labels_

        for i in range(r):
            for j in range(i,r):
                if labels[i] == labels[j]:
                    M[i][j] += 1
        
    M = M + M.T
    M /= N

    cutplot = np.zeros((N+1 ,2), dtype='f2<')
    for l in xrange(N+1):
        # Construct graph G for which M[i][j] > l/N
        G = M > 1.0 * l /N
        
        n_comp, labels = cs.cs_graph_components(G)
        cutplot[l][0] = l * 1.0 /N
        cutplot[l][1] = n_comp
    return M, cutplot


def find_cutoff(cutplot):
    """
    Finds the weight cutoff based on the cutplot.
    """
    from itertools import groupby

    group = groupby(cutplot[:,1])
    val = max(group, key=lambda k: len(list(k[1])))[0]
    
    for c in cutplot:
        if c[1] == val:
            return c[0] + 0.1


def category_mat(S, M, cutplot, C=None):
    """
    Predicts the category for each data point
    """ 
    if C == None:
        C = find_cutoff(cutplot)
    # Build a new graph on S with edges M[i][j] > C
    newG = M > C
    n_comp, labels = cs.cs_graph_components(newG)
    
    return labels
   
