import numpy as np
from scipy.sparse import coo_matrix, issparse



def KL_divergence(p, q, normalize=True):
    """ 
    Compute KL divergence of distribution vector p and 
    each row of distribution matrix Q, K(p || q) for q in Q.
    
    Parameters
    ----------
    p : 2-dim floating point array
        The distribution with which KL_divergence is computed.
        2-dim array must has the form (1,n).
    q : 2-dim floating point array
        Matrix containing distributions to be compared with `p`
    normalize : Logical
        normalize p and q if True. 
    """
    #Can we use a matrix for p?
#    indices = np.indices((len(p),len(q)))
#    logp = np.log2(p[indices[0]]/q[indices[1]])
#    out  = np.einsum('ik,ijk->ij',p,logp)
#    return out
    if normalize:
        p = row_normalize(p, norm='sum')
        q = row_normalize(q, norm='sum')

    old = np.seterr(divide='ignore') # Suppress division by zero errors
    logp = np.log2(p/q)
    np.seterr(**old) # Restore error settings
    KLD  = np.dot(logp, p.T)
    KLD  = np.ravel(KLD)
    return KLD



def JS_divergence(p, q, normalize=False, metric=True):
    """  
    Compute (the square root of) the Jensen-Shannon divergence 
    of two vectors, defined by
       JSD = (KL(p || m) + KL(q || m))/2
    where m = (p+q)/2. 
    The square root of the JS divergence is a metric.

    Parameters
    ----------
    p : 1-dim floating point array
        First distribution.
    q : 1-dim floating point array
        Second distribution.
    norms : Logical
    """
    m   = (p+q)/2
    JSD = (KL_divergence(p, m, normalize) + KL_divergence(q, m, normalize))/2 

    if metric:
        JSD = JSD**0.5

    return JSD



def JS_dismat(rows, mat, norms=None, fill_tril=True):
    """
    Compute the distance matrix for a set of distributions in `mat` 
    by computing pairwise Jansen-Shannon divergences.
    
    Parameters
    ----------
    rows : 1-dim array
        Species distributions whose distances are to be calculated.
    mat : 2-dim floating point array
        The set of probability distributions where each row is a 
        distribution.
    norms : normalize mat if none
    """
    mat = mat[rows]

    if norms is None:
        mat = row_normalize(mat, norm='sum')        

    dsm = np.zeros((len(rows), len(rows)), dtype=np.float64)
    indices = np.triu_indices_from(dsm)
    f = np.vectorize(lambda i, j: JS_divergence(np.atleast_2d(mat[i,:]), 
                                                np.atleast_2d(mat[j,:]), norms))
    dsm[indices] = f(*indices)[:]

    if fill_tril:
        indices = np.tril_indices_from(dsm, -1)
        dsm[indices] = dsm.T[indices]

    return dsm



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



def row_normalize(m, norm='SS'):
    """
    Takes a 2-d array and returns it with its rows normalized.

    Parameters
    ----------
    norm : str
        Method to be used as a normalizing constant. 'SS' is the 
        rooted square sum (yields a unit vector) while 'sum' uses
        an ordinary summation (yields a probability).
    """
    if norm=='SS':
        norms = row_norms(m)
    elif norm=='sum':
        norms = m.sum(axis=1)
    else:
        raise Exception('Invalid normalizing parameter.')
    return m / norms[:, np.newaxis]



def rand_pt_unit_sphere(n, seed=None):
    """
    """
    np.random.seed(seed)

    pt = np.random.normal(size=n)
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


def row_acos(row, matrix, norms=None):
    
    cosines = row_cosines(row, matrix, norms=norms)

    # To optimize
    for i in xrange(len(cosines)):
        if np.allclose(cosines[i], 1):
            cosines[i] = 1
        if np.allclose(cosines[i], -1):
            cosines[i] = -1

    return np.arccos(cosines)
    
    
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


def row_acos_mat(rows, mat, norms=None, fill_tril=True):

    cos_mat = row_cos_mat(rows, mat, norms=norms, fill_tril=fill_tril)

    # To optimize
    for i in np.ndindex(*cos_mat.shape):
        if np.allclose(cos_mat[i], 1):
            cos_mat[i] = 1
        if np.allclose(cos_mat[i], -1):
            cos_mat[i] = -1

    return np.arccos(cos_mat)

    
def hstack_coo(mat_ls):
    """
    """
    #TODO: Raise an exception if matrices do not all have the same
    #number of rows
    if len(mat_ls) > 0:
        shape_0 = mat_ls[0].shape[0]
    shape_1 = np.sum([mat.shape[1] for mat in mat_ls]) 

    data = np.hstack([mat.data for mat in mat_ls])
    row = np.hstack([mat.row for mat in mat_ls])

    col_ls = [m.col.copy() for m in mat_ls]
    offset = 0
    for i in xrange(len(mat_ls)):
        col_ls[i] += offset
        offset += mat_ls[i].shape[1]
    col = np.hstack(col_ls)

    return coo_matrix((data, (row, col)),
                      shape=(shape_0, shape_1),
                      dtype=data.dtype)


def count_matrix(arr, slices, m=None):
    """
    arr : numpy array
    slices : list of slices
    m : integer
    """
    if not m:
        m = arr.max()
    shape = (m, len(slices))

    data = np.ones_like(arr)
    row_indices = arr
    col_indices = np.empty_like(arr)
    for i,s in enumerate(slices):
        col_indices[s] = i

    return coo_matrix((data, (row_indices, col_indices)),
                      shape=shape, dtype=np.int32)

    

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


def hstack_coo_test():

    dense_mat_ls = [np.random.random((3,4)),
                    np.random.random((3,5)),
                    np.random.random((3,6))]

    mat_ls = [coo_matrix(m) for m in dense_mat_ls]
    
    assert (np.hstack(dense_mat_ls) == hstack_coo(mat_ls).toarray()).all()


def count_matrix_test():

    arr = [1, 2, 4, 2, 1]
    slices = [slice(0,1), slice(1, 3), slice(3,3), slice(3, 5)]
    m = 6
    result = coo_matrix([[0, 0, 0, 0],
                         [1, 0, 0, 1],
                         [0, 1, 0, 1],
                         [0, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 0]])
    
    assert (result.toarray() == count_matrix(arr, slices, m).toarray()).all()
