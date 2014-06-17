import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, issparse



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



#
# Functions for computing angular distances between points projected
# onto an n-sphere
# 

def row_norms(matrix):
    """
    """
    norms = np.empty(matrix.shape[0], dtype=matrix.dtype)
    for i in xrange(norms.shape[0]):
        row = matrix[i:i+1, :]
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


def angle_sparse(P, Q):
    """
    P and Q are sparse matrices from scipy.sparse.

    Angles are computed wrt the rows of P and wrt the columns of Q.
    """
    P = P.tocsr()
    Q = Q.tocsc()

    # Normalize P row-wise and Q column-wise
    P_inv_norms = 1 / np.sqrt(P.multiply(P).sum(1))
    Q_inv_norms = 1 / np.sqrt(Q.multiply(Q).sum(0))

    P = P.multiply(csc_matrix(P_inv_norms))
    Q = Q.multiply(csr_matrix(Q_inv_norms))

    cos_PQ = P.dot(Q).toarray()
    out = np.arccos(cos_PQ)

    out = np.squeeze(out)
    if out.ndim == 0:
        return out[()]
    return out


def angle(P, Q):
    """
    P and Q are arrays of either dimension 1 or 2.

    If P is a matrix, then angles are computed wrt the rows of P. If Q
    is a matrix, then angles are computed wrt the columns of Q.

    """
    if P.ndim < 2:
        P = P.reshape((1,P.size))
    if Q.ndim < 2:
        Q = Q.reshape((Q.size,1))

    # Normalize P row-wise and Q column-wise
    P /= np.sqrt((P*P).sum(1)[:,np.newaxis])
    Q /= np.sqrt((Q*Q).sum(0)[np.newaxis,:])

    cos_PQ = np.dot(P,Q)
    out = np.arccos(cos_PQ)

    out = np.squeeze(out)
    if out.ndim == 0:
        return out[()]
    return out


#
# Functions for computing information theoretic distances
# 

def H(P):
    """
    P is an array of dimension 1 or 2.

    If P is a matrix, entropy is computed row-wise. (P is assumed to
    be left stochastic)
    """
    if P.ndim < 2:
        P = P.reshape((1,P.size))

    logP = np.log2(P)
    logP[(P == 0)] = 0.
    out = -1 * (P * logP).sum(1)

    out = np.squeeze(out)
    if out.ndim == 0:
        return out[()]
    return out


def cross_H(P,Q):
    """
    P and Q are arrays of either dimension 1 or 2.

    If P is a matrix, then cross entropy is computed row-wise on P. (P
    is assumed to be left stochastic.) If Q is a matrix, cross entropy
    is computed column-wise on Q. (Q is assumed to be right
    stochastic.)
    """
    if P.ndim < 2:
        P = P.reshape((1,P.size))
    if Q.ndim < 2:
        Q = Q.reshape((Q.size,1))

    P_ = np.tile(P.reshape(P.shape[0], P.shape[1], 1), (1,1,Q.shape[1]))
    Q_ = np.tile(Q.reshape(1, Q.shape[0], Q.shape[1]), (P.shape[0],1,1))
    logQ_ = np.tile(np.log2(Q.reshape(1, Q.shape[0], Q.shape[1])), 
                    (P.shape[0],1,1))
    out = (P_ * logQ_)

    P_zeros = (P_ == 0)
    Q_zeros = (Q_ == 0)
    PQ_zeros = np.bitwise_and(P_zeros, Q_zeros)
    out[PQ_zeros] = 0.

    out = np.squeeze(-1 * out.sum(1))
    if out.ndim == 0:
        return out[()]
    return out


def KL_div(P, Q):
    """
    P and Q are arrays of either dimension 1 or 2.

    If P is a matrix, then KL divergence is computed row-wise on P. (P
    is assumed to be left stochastic.) If Q is a matrix, KL divergence
    is computed column-wise on Q. (Q is assumed to be right
    stochastic.)
    """
    if P.ndim < 2:
        return np.squeeze(cross_H(P, Q) - H(P))
    out = np.squeeze(cross_H(P, Q) - H(P)[:,np.newaxis])
    if out.ndim == 0:
        return out[()]
    return out


def JS_div(P, Q, metric=True):
    """
    P and Q are arrays of either dimension 1 or 2.

    If P is a matrix, then JS divergence is computed row-wise on P. (P
    is assumed to be left stochastic.) If Q is a matrix, JS divergence
    is computed column-wise on Q. (Q is assumed to be right
    stochastic.)
    
    The square root of the JS divergence is a metric.
    """
    if P.ndim < 2:
        P = P.reshape((1,P.size))
    if Q.ndim < 2:
        Q = Q.reshape((Q.size,1))
    
    P_ = np.tile(P[:,:,np.newaxis], (1,1,Q.shape[1]))
    Q_ = np.tile(Q[np.newaxis,:,:], (P.shape[0],1,1))
    M_ = .5 * (P_ + Q_)
    logM_ = np.log2(M_)

    P_zeros = (P_ == 0)
    Q_zeros = (Q_ == 0)
    PQ_zeros = np.bitwise_and(P_zeros, Q_zeros)
    logM_[PQ_zeros] = 0.

    HP = H(P)
    HQ = H(Q.T)

    PM_KLdiv = -1 * (P_ * logM_).sum(1) - HP.reshape(HP.size,1)
    QM_KLdiv = -1 * (Q_ * logM_).sum(1) - HQ.reshape(1,HQ.size) 

    out = .5 * (PM_KLdiv + QM_KLdiv)

    if metric:
        out = np.sqrt(out)

    out = np.squeeze(out)
    if out.ndim == 0:
        return out[()]
    return out



# def row_kld(row, mat, normalize=True, norms=None):
#     """ 
#     Compute KL divergence of distribution vector `row` and 
#     each row of distribution matrix `mat`.
    
#     Parameters
#     ----------
#     row : 2-dim floating point array
#         The distribution with which KL divergence is computed.
#         2-dim array must has the form (1,n).
#     mat : 2-dim floating point array
#         Matrix containing distributions to be compared with `row`
#     normalize : Logical
#         Normalizes `row` and `mat` if None. 
#     """
#     if normalize:
#         row = row_normalize(row, norm='sum')
#         mat = row_normalize(mat, norm='sum')

#     old = np.seterr(divide='ignore') # Suppress division by zero errors
#     logp = np.log2(row/mat)
#     np.seterr(**old)                 # Restore error settings
#     logp[np.isinf(logp)] = 0         # replace inf with zeros
#     KLD  = np.dot(logp, row.T)
#     KLD  = np.ravel(KLD)
#     return KLD


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


def log_dot(v):
    """
    computes dot product of a vector v and its log
    """
    return np.dot(v, np.log2(v))


def posterior(row, mat, norms=None):
    """
    Compute weighted average of posterors used in sim_top_doc
    """
    row = row_normalize(row, norm='sum')    # weights must add up to one
    post = row_normalize(mat, norm='sum')
    post /= post.sum(axis=0)
    out = (row * post).sum(axis=1)
    
    return out 
