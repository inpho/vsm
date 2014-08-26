import numpy as np

def gen_colors(clusters):
    """
    Takes 'clusters' and creates a list of colors so a cluster has a color.
    
    :param clusters: A flat list of integers where an integer represents which
        cluster the information belongs to.
    :type clusters: list
    
    :returns: colorm : list
        A list of colors obtained from matplotlib colormap cm.hsv. The
        length of 'colorm' is the same as the number of distinct
        clusters.
    """
    import matplotlib.cm as cm
    
    n = len(set(clusters))
    colorm = [cm.hsv(i * 1.0 /n, 1) for i in xrange(n)]
    return colorm


def plot_clusters(arr, labels, clusters=[], size=[]):
    """	
    Takes 2-dimensional array(simmat), list of clusters, list of labels,
    and list of marker size. 'clusters' should be a flat list which can be
    obtained from cluster_topics(by_cluster=False).
    Plots each clusters in different colors.
    
    :type arr: 2-dimensional array
    :param arr: Array has x, y coordinates to be plotted on a 2-dimensional
        space.
    
    :param labels: List of labels to be displayed in the graph. 
    :type labels: list
    
    :param clusters: A flat list of integers where an integer represents which
        cluster the information belongs to. If not given, it returns a
        basic plot with no color variation. Default is an empty list.
    :type clusters: list, optional
    
    :param size: List of markersize for points where markersize can note the
        importance of the point. If not given, 'size' is a list of
        fixed markersize, 40. Default is an empty list.
    :type size: list, optional

    :returns: plt : maplotlit.pyplot object
        A graph with scatter plots from 'arr'.
    """
    import matplotlib.pyplot as plt

    n = arr.shape[0]
    X = arr[:,0]
    Y = arr[:,1]

    if len(size) == 0:
        size = [40 for i in xrange(n)]
        
    fig = plt.figure(figsize=(10,10))
    ax = plt.subplot(111)

    if len(clusters) == 0:
        plt.scatter(X, Y, size)
    else:	
        colors = gen_colors(clusters)
        colors = [colors[i] for i in clusters]

        for i in xrange(n):
            plt.scatter(X[i], Y[i], size, color=colors[i])

    ax.set_xlim(np.min(X) - .1, np.max(X) + .1)
    ax.set_ylim(np.min(Y) - .1, np.max(Y) + .1)
    ax.set_xticks([])
    ax.set_yticks([])

    for label, x, y in zip(labels, X, Y):
        plt.annotate(label, xy = (x, y), xytext=(-2, 3), 
                     textcoords='offset points', fontsize=10)

    plt.show()
