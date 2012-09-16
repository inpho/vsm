import numpy as np
from scipy import sparse


def row_norms(matrix):

    norms = np.empty(matrix.shape[0])

    sp = sparse.issparse(matrix)

    if sp:

        matrix = matrix.tocsr()

    for i in xrange(norms.shape[0]):

        row = matrix[i:i+1, :]

        if sp:
            
            row = row.toarray()

        norms[i] = np.dot(row, row.T)**0.5
        
    return norms



def col_norms(matrix):

    return row_norms(matrix.T)



def row_normalize(m):
    """
    Takes a 2-d array and returns it with its rows normalized
    """
    norms = row_norms(m)

    return m / norms[:, np.newaxis]



def rand_pt_unit_sphere(n, seed=None):

    np.random.seed(seed)

    pt = np.random.random(n)

    pt = pt * 2 - 1

    pt = pt / np.dot(pt, pt)**.5

    return pt



def naive_cconv(v, w):

    out = np.empty_like(v)

    for i in xrange(v.shape[0]):

        out[i] = np.dot(v, np.roll(w[::-1], i + 1))

    return out



#
# Testing
#



def test_row_normalize():

    m = np.random.random((5,7))

    m = row_normalize(m)

    print np.allclose(row_norms(m), np.ones(5))
