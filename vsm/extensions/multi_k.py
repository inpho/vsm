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

def multi_k(S, N=30, D_max=30, C=0.5):
    import matplotlib.pyplot as plt

    M = connection_matrix(S=S, N=N, D_max=D_max)
    c = cutplot(M=M, N=N)

    x, y = zip(*c)
    plt.plot(x,y)
#    plt.show()

    return plt, c


def connection_matrix(S, N=10, D_max=30):
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


def count_x(x, ls):
    """
    Counts the number of occurrences of `x` in list `ls`.
    """
    count = 0
    for l in ls:
        if l == x:
            count += 1

    return count

def find_y(x, plot):
    """
    Given the x value, finds the y value.
    plot is a list with tuples (x,y).
    """
    for i,j in plot:
        if x == i:
            return j
    # if x is not found in plot
    return 'val not found.'


def refine(S, M, cutplot, N=10, C=0.5):
    
    cutplot_ = []
    for i in xrange(len(cutplot)):
        # !!! ERROR may be the wrong input, x.
        # never hits the if < x < with these inputs.
        x = cutplot[i][0]

        for l in xrange(N + 1):
            if 1.0 * l/N < x and x < 1.0 * (l + 1)/N:
                # refine cut plot function.
                it = 1.0 * l/N
                y = find_y(it, cutplot)
                cutplot_.append((x, y))
        else:
            cutplot_.append((x, cutplot[i][1]))

    # Build a new graph on S with edges M[i][j] > C
    newG = M > C
    
    categorize = []
    n_comp, labels = cs.cs_graph_components(newG)
    
    for i in xrange(len(S)):
        count = count_x(labels[i], labels)
        categorize.append((S[i], count))

    return categorize, cutplot_


