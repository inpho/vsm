from collections import defaultdict
from glob import glob
from itertools import islice
from numpy import argmin, argsort, finfo, reshape, size, sum, unique, vstack, zeros
from utils import *
from corpus import *

def aggregate(regexp, col=-1, header=0):

    files = glob(regexp)

    tmp = defaultdict(list)

    for statefile in files:
        for n, row in enumerate(islice(open(statefile), header, None)):
            tmp[n].append(row.split()[col])

    S = len(files)
    N = len(tmp)

    values = zeros((N, S), dtype=int)

    for n in xrange(N):
        values[n, :] = tmp[n]

    if S == 1:
        values = reshape(values, N)

    return values

def count_unique_values(values):

    tmp = zeros(len(values), dtype=int)

    for n, v in enumerate(values):
        tmp[n] = len(unique(v))

    return tmp

def get_empirical_phi(statefile):

    nwt = get_nwt(statefile)
    phi = nwt / (sum(nwt, 0).astype('float') + finfo('float').eps)

    return phi

def get_empirical_theta(statefile):

    ntd = get_ntd(statefile)
    theta = ntd / (sum(ntd, 0).astype('float') + finfo('float').eps)

    return theta

def get_nwt(statefile):

    w_idx = aggregate(statefile, col=2)
    t_idx = aggregate(statefile)

    return get_2d_counts(vstack((w_idx, t_idx)).T)

def get_ntd(statefile):

    t_idx = aggregate(statefile)
    d_idx = aggregate(statefile, col=0)

    return get_2d_counts(vstack((t_idx, d_idx)).T)

def get_nw(statefile):

    w_idx = aggregate(statefile, col=2)

    return get_1d_counts(w_idx)

def get_nt(statefile):

    t_idx = aggregate(statefile)

    return get_1d_counts(t_idx)

def get_1d_counts(data):

    tmp = defaultdict(int)

    for i in data:
        tmp[i] += 1

    I = len(tmp)

    counts = zeros(I, dtype=int)

    for i in xrange(I):
        counts[i] = tmp[i]

    return counts

def get_2d_counts(data):

    tmp = defaultdict(int)

    I = J = 1

    for x in data:

        i, j = x[0], x[1]
        I, J = max(I, i+1), max(J, j+1)

        tmp[(i, j)] += 1

    counts = zeros((I, J), dtype=int)

    for i in xrange(I):
        for j in xrange(J):
            counts[i, j] = tmp[(i, j)]

    return counts

def align_topics(statefile1, statefile2):

    phi1 = get_empirical_phi(statefile1)
    phi2 = get_empirical_phi(statefile2)

    T1, T2 = size(phi1, 1), size(phi2, 1)

    dist = zeros((T1, T2))

    for i in xrange(T1):
        for j in xrange(T2):
            dist[i, j] = hellinger(phi1[:, i], phi2[:, j])

    return dist

def get_corpus(statefile, header=0):

    corpus = Corpus()

    name = None
    tokens = []

    for row in islice(open(statefile), header, None):

        fields = row.split()

        if fields[1] == '0':

            if name and tokens:
                corpus.add(name, tokens)

            name = fields[0]
            tokens = []

        tokens.append(fields[3])

    if name and tokens:
        corpus.add(name, tokens)

    return corpus
