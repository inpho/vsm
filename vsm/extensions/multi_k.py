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

def multi_k(S, N=30, D=(10,30), C=0.5):
    import matplotlib.pyplot as plt

    M = connection_matrix(S=S, N=N, D=D)
    c = cutplot(M=M, N=N)
    category_func = categories(S, M, c, C=C)

    x, y = zip(*c)
    plt.plot(x,y)
#    plt.show()

    return plt, c, category_func


def connection_matrix(S, N=10, D=(10,30)):
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
                    M[j][i] += 1
                    
    # normalize M[i][j] 
    M /= N

    return M



def cutplot(M, N=10):
    # Build Weight matrix, W. 
    # Q. what's the purpose of the Weight matrix?
    W = M
    cutplot = []
    for l in xrange(N+1):
        # Construct graph G for which M[i][j] > l/N
        G = M > 1.0 * l /N
        
        n_comp, labels = cs.cs_graph_components(G)
        cutplot.append((l * 1.0 /N, n_comp))
    return cutplot


def categories(S, M, cutplot, C=0.5):
    """
    Returns a dictionary with indices in S as keys
    and category labels as values.
    """ 
    # Build a new graph on S with edges M[i][j] > C
    newG = M > C
    n_comp, labels = cs.cs_graph_components(newG)
    
    categories = {}
    for i in xrange(len(S)):
        categories[i] = labels[i]

    return categories 
   
