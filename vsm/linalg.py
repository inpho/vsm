import numpy as np
from scipy.sparse import coo_matrix, issparse



def KL_divergence(p, q):
    """ 
    Compute KL divergence of distribution vector p and 
    each row of distribution matrix Q, K(p || q) for q in Q.
    """
    #Can we use a matrix for p?
#    indices = np.indices((len(p),len(q)))
#    logp = np.log2(p[indices[0]]/q[indices[1]])
#    out  = np.einsum('ik,ijk->ij',p,logp)
#    return out
    logp = np.log2(p/q)
    return np.dot(logp, p)



def JS_divergence(p, q, metric=True):
    """  
    Compute (the square root of) the Jensen-Shannon divergence 
    of two vectors, defined by
       JSD = (KL(p || m) + KL(q || m))/2
    where m = (p+q)/2. 
    The square root of the JS divergence is a metric.
    """
    m   = (p+q)/2
    JSD = (KL_divergence(p, m) + KL_divergence(q, m))/2 

    if metric:
        JSD = JSD**0.5

    return JSD



def JS_dismat(P, fill_tril=True):
    """
    Compute the distance matrix for set of distributions P by computing 
    pairwise Jansen-Shannon divergences.
    """
    # Need to replace it with a faster way
    dismat = np.zeros((P.shape[0], P.shape[0]))
    for i,j in zip(*np.triu_indices_from(dismat, k=1)):
        dismat[i,j] = JS_divergence(P[i,:], P[j,:])

    if fill_tril:
        indices = np.tril_indices_from(dismat, -1)
        dismat[indices] = dismat.T[indices]

    return dismat



def row_norms(matrix):
    """
    """
    if issparse(matrix):
        return row_norms_sparse(matrix)

    norms = np.empty(matrix.shape[0], dtype=matrix.dtype)
    for i in xrange(norms.shape[0]):
        row = matrix[i:i+1, :]
        norms[i] = np.dot(row, row.T)**0.5
        
    return norms



def row_norms_sparse(matrix):
    """
    """
    norms = np.empty(matrix.shape[0], dtype=matrix.dtype)
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

    old = np.seterr(divide='ignore', invalid='ignore') # Suppress division by zero errors
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

