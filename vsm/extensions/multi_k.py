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

def multi_k(samples, n=30, distr=(10,30), cutoff=None, n_cls=None):
    """
    Oveall procedure run. Returns cutplot and category_func.
    """
    
    import matplotlib.pyplot as plt

    mat, cutplot = connection_matrix(samples=samples, n=n, distr=distr)

    if cutoff == None:
        cutoff = find_cutoff(cutplot, n_cls=n_cls)
    print "Weight cutoff is set to ", cutoff
    category_func = category_mat(samples, mat, cutplot, cutoff=cutoff)

    x, y = zip(*cutplot)
    plt.plot(x,y)

    return plt, cutplot, category_func


def connection_matrix(samples, n=10, distr=(10,30)):
    """
    samples : set S of r points in Rn.
    n : number n of clusterings to perform.
    distr : numbers of clusters.
        <- this should be a distribution over 0,...,n. 
        Let us use a uniform integer for a moment. 
    cutoff : weight cutoff, 0 <= cutoff <= 1.
    
    could make n,distr,cutoff optional by setting default values.
    
    :returns: mat : connection matrix 
             cutplot :  A cut plot
                f : [0,1] -> Z
    """
    r = len(samples)
    mat = np.zeros((r, r))
    
    for n_ in xrange(n):
        # select integer d from distr[0] to distr[1] 
        d = randrange(distr[0],distr[1]+1)
        km = KMeans(n_clusters=d, init='k-means++',
                    max_iter=100, n_init=1,verbose=False)
        km.fit(samples)
        labels = km.labels_

        for i in range(r):
            for j in range(i,r):
                if labels[i] == labels[j]:
                    mat[i][j] += 1
        
    mat = mat + mat.T
    mat /= n

    cutplot = np.zeros((n+1 ,2), dtype='f2<')
    for l in xrange(n+1):
        # Construct graph for which mat[i][j] > l/n
        graph = mat > 1.0 * l /n
        
        n_comp, labels = cs.cs_graph_components(graph)
        cutplot[l][0] = l * 1.0 /n
        cutplot[l][1] = n_comp
    return mat, cutplot


def find_cutoff(cutplot, n_cls=None):
    """
    Finds the weight cutoff based on the longest run in cutplot.
    If n_cls is provided, finds the cutoff point where n_cls
    clusters are formed."""
    from itertools import groupby

    if n_cls != None:
        for c in cutplot:
            if c[1] == n_cls:
                return c[0]

    group = groupby(cutplot[:,1])
    val = max(group, key=lambda k: len(list(k[1])))[0]
    
    for c in cutplot:
        if c[1] == val:
            return c[0] + 0.01


def category_mat(samples, mat, cutplot, cutoff=None):
    """
    Predicts the category for each data point
    """ 
    if cutoff == None:
        cutoff = find_cutoff(cutplot)
    # Build a new graph on samples with edges mat[i][j] > cutoff
    newG = mat > cutoff
    n_comp, labels = cs.cs_graph_components(newG)
    
    return labels
   
