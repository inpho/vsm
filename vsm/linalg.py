import numpy as np
from scipy.sparse import coo_matrix, issparse




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



def log_dot(v):
    """
    computes dot product of a vector v and its log
    """
    return np.dot(v, np.log2(v))



#
# Methods to calculate a simulatiry vector
# 

def row_cosines(row, mat, norms=None, normalize=True):
    """
    `row` must be a 2-dimensional array.
    """
    if issparse(mat):
        mat = mat.tocsr()
        nums = sparse_mvdot(mat, row.T)
    else:
        nums = np.dot(mat, row.T)
        nums = np.ravel(nums)

    if norms is None:
        norms = row_norms(mat)

    row_norm = row_norms(row)[0]
    dens = norms * row_norm

    old = np.seterr(divide='ignore', invalid='ignore') # Suppress division by zero errors
    out = nums / dens
    np.seterr(**old) # Restore error settings

    return out


def row_acos(row, mat, norms=None):
    
    cosines = row_cosines(row, mat, norms=norms)

    # To optimize
    for i in xrange(len(cosines)):
        if np.allclose(cosines[i], 1):
            cosines[i] = 1
        if np.allclose(cosines[i], -1):
            cosines[i] = -1

    return np.arccos(cosines)



def row_kld(row, mat, normalize=True, norms=None):
    """ 
    Compute KL divergence of distribution vector `row` and 
    each row of distribution matrix `mat`.
    
    Parameters
    ----------
    row : 2-dim floating point array
        The distribution with which KL divergence is computed.
        2-dim array must has the form (1,n).
    mat : 2-dim floating point array
        Matrix containing distributions to be compared with `row`
    normalize : Logical
        Normalizes `row` and `mat` if None. 
    """
    if normalize:
        row = row_normalize(row, norm='sum')
        mat = row_normalize(mat, norm='sum')

    old = np.seterr(divide='ignore') # Suppress division by zero errors
    logp = np.log2(row/mat)
    np.seterr(**old)                 # Restore error settings
    logp[np.isinf(logp)] = 0         # replace inf with zeros
    KLD  = np.dot(logp, row.T)
    KLD  = np.ravel(KLD)
    return KLD



def posterior(row, mat, norms=None):
    """
    Compute weighted average of posterors used in sim_top_doc
    """
    row = row_normalize(row, norm='sum')    # weights must add up to one
    post = row_normalize(mat, norm='sum')
    post /= post.sum(axis=0)
    out = (row * post).sum(axis=1)
    
    return out 



#
# Methods to calculate a (dis)similarity matrix
#    
    
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



def row_js_mat(rows, mat, norms=None, fill_tril=True):
    """
    Compute a distance matrix for a set of distributions in `mat` 
    by computing pairwise Jansen-Shannon divergences.
    
    Parameters
    ----------
    rows : 1-dim array
        Specifies distributions whose distances are to be calculated.
    mat : 2-dim floating point array
        The set of probability distributions where each row is a 
        distribution.
    norms : normalize mat if none
    fill_tril : Dummy
        Not used (row_js_mat always returns a symmetric matrix)
    """
    # Known issue: Some zero entories (diagonal) get nonzero scores 
    # due to rounding
    P = mat[rows]

    if norms is None:
        P = row_normalize(P, norm='sum')        

    # Compute midpoint part
    M = np.zeros((len(rows), len(rows)), dtype=np.float64)
    indices = np.triu_indices_from(M)
    f = np.vectorize(lambda i, j: log_dot(P[i,:]+P[j,:])-2) 
    M[indices] = f(*indices)[:]
    # Make it symetric
    indices = np.tril_indices_from(M, -1)
    M[indices] = M.T[indices]

    # Compute probability part
    P = np.tile((P*np.log2(P)).sum(axis=1), (len(rows),1))
    P = P + P.T

    out = ((P-M)/2).clip(0) 

    return out**0.5


    
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

