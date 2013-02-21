import numpy as np
from scipy.sparse import issparse



def row_norms(matrix):
    """
    """
    if issparse(matrix):
        return row_norms_sparse(matrix)

    norms = np.empty(matrix.shape[0])
    for i in xrange(norms.shape[0]):
        row = matrix[i:i+1, :]
        norms[i] = np.dot(row, row.T)**0.5
        
    return norms



def row_norms_sparse(matrix):
    """
    """
    norms = np.empty(matrix.shape[0])
    matrix = matrix.tocsr()
    for i in xrange(norms.shape[0]):
        row = matrix[i:i+1, :].toarray()
        norms[i] = np.dot(row, row.T)**0.5

    return norms



def row_normalize(m):
    """
    Takes a 2-d array and returns it with its rows normalized
    """
    norms = row_norms(m)
    return m / norms[:, np.newaxis]



def rand_pt_unit_sphere(n, seed=None):
    """
    """
    np.random.seed(seed)

    pt = np.random.random(n)
    pt = pt * 2 - 1
    pt = pt / np.dot(pt, pt)**.5

    return pt



def naive_cconv(v, w):
    """
    """
    out = np.empty_like(v)
    for i in xrange(v.shape[0]):
        out[i] = np.dot(v, np.roll(w[::-1], i + 1))

    return out



def sparse_mvdot(m, v, submat_size=10000):
    """
    For sparse matrices. The expectation is that a dense view of the
    entire matrix is too large. So the matrix is split into
    submatrices (horizontal slices) which are converted to dense
    matrices one at a time.
    """

    m = m.tocsr()
    if issparse(v):
        v = v.toarray()
    v = v.reshape((v.size, 1))
    out = np.empty((m.shape[0], 1))

    if submat_size < m.shape[1]:
        print 'Note: specified submatrix size is '\
              'less than the number of columns in matrix'
        m_rows = 1
        k_submats = m.shape[0]
    elif submat_size > m.size:
        m_rows = m.shape[0]
        k_submats = 1
    else:
        m_rows = int(submat_size / m.shape[1])
        k_submats = int(m.shape[0] / m_rows)
    for i in xrange(k_submats):
        i *= m_rows
        j = i + m_rows
        submat = m[i:j, :]
        out[i:j, :] = np.dot(submat.toarray(), v)
    if j < m.shape[0]:
        submat = m[j:, :]
        out[j:, :] = np.dot(submat.toarray(), v)

    return np.squeeze(out)



def row_cosines(row, matrix, norms=None):
    """
    `row` must be a 2-dimensional array.
    """
    if issparse(matrix):
        matrix = matrix.tocsr()
        nums = sparse_mvdot(matrix, row.T)
    else:
        nums = np.dot(matrix, row.T)
        nums = np.ravel(nums)

    if norms is None:
        norms = row_norms(matrix)

    row_norm = row_norms(row)[0]
    dens = norms * row_norm

    old = np.seterr(divide='ignore') # Suppress division by zero errors
    out = nums / dens
    np.seterr(**old) # Restore error settings

    return out



def row_cos_mat(rows, mat, norms=None, fill_tril=True):

    if issparse(mat):
        mat = mat.tocsr()[rows].toarray()
    else:
        mat = mat[rows]

    if not norms:
        norms = row_norms(mat)
    else:
        norms[rows]

    sm = np.zeros((len(rows), len(rows)), dtype=np.float64)
    indices = np.triu_indices_from(sm)
    f = np.vectorize(lambda i, j: (np.dot(mat[i,:], mat[j,:].T) /
                                   (norms[i] * norms[j])))
    sm[indices] = f(*indices)[:]

    if fill_tril:
        indices = np.tril_indices_from(sm, -1)
        sm[indices] += sm.T[indices]

    return sm


#
# Testing
#


def test_row_normalize():

    m = np.random.random((5,7))
    m = row_normalize(m)

    assert np.allclose(row_norms(m), np.ones(5))



def test_row_cos_mat():

    m = np.random.random((10,5))
    out_1 = np.zeros((10,10))
    for i, j in zip(*np.triu_indices_from(out_1)):
        out_1[i, j] = (np.dot(m[i], m[j])
                       / (np.dot(m[i], m[i])**.5
                          * np.dot(m[j], m[j])**.5))
    out_2 = row_cos_mat(range(10), m, fill_tril=False)
        
    assert np.allclose(out_1, out_2), (out_1, out_2)
