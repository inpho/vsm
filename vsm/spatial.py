import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, issparse



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


def scipy_cdist(**kwargs):
    """
    Returns a wrapper of `scipy.spatial.distance.cdist`.

    P and Q are arrays of either dimension 1 or 2.

    If P is a matrix, then angles are computed wrt the rows of P. If Q
    is a matrix, then angles are computed wrt the columns of Q.
    """
    from scipy.spatial.distance import cdist

    def dist_fn(P, Q):

        P = P.astype(np.float)
        Q = Q.astype(np.float)

        if P.ndim < 2:
            P = P.reshape((1,P.size))
        if Q.ndim < 2:
            Q = Q.reshape((Q.size,1))
            
        out = cdist(P, Q.T, **kwargs)

        out = np.squeeze(out)
        if out.ndim == 0:
            return out[()]
        return out

    return dist_fn


#
# Functions for computing angular distances between points projected
# onto an n-sphere
# 


def angle(P, Q):
    """
    P and Q are arrays of either dimension 1 or 2.

    If P is a matrix, then angles are computed wrt the rows of P. If Q
    is a matrix, then angles are computed wrt the columns of Q.

    """
    P = P.astype(np.float)
    Q = Q.astype(np.float)

    if P.ndim < 2:
        P = P.reshape((1,P.size))
    if Q.ndim < 2:
        Q = Q.reshape((Q.size,1))

    # Normalize P row-wise and Q column-wise
    P /= np.sqrt((P*P).sum(1)[:,np.newaxis])
    Q /= np.sqrt((Q*Q).sum(0)[np.newaxis,:])

    cos_PQ = np.dot(P,Q)
    
    # Rounding errors may result in values outside of the domain of
    # inverse cosine.
    cos_PQ[(cos_PQ > 1)] = 1.
    cos_PQ[(cos_PQ < -1)] = -1.

    out = np.arccos(cos_PQ)

    out = np.squeeze(out)
    if out.ndim == 0:
        return out[()]
    return out


def angle_sparse(P, Q):
    """
    P and Q are sparse matrices from scipy.sparse.

    Angles are computed wrt the rows of P and wrt the columns of Q.
    """
    P = P.tocsc().astype(np.float)
    Q = Q.tocsr().astype(np.float)

    # Normalize P row-wise and Q column-wise
    P_inv_norms = 1 / np.sqrt(P.multiply(P).sum(1))
    Q_inv_norms = 1 / np.sqrt(Q.multiply(Q).sum(0))

    # Requires scipy version >= 0.13
    P = P.multiply(csc_matrix(P_inv_norms))
    Q = Q.multiply(csr_matrix(Q_inv_norms))

    P = P.tocsr()
    Q = Q.tocsc()
    cos_PQ = P.dot(Q).toarray()

    # Rounding errors may result in values outside of the domain of
    # inverse cosine.
    cos_PQ[(cos_PQ > 1)] = 1.
    cos_PQ[(cos_PQ < -1)] = -1.

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
    P = P.astype(np.float)

    if P.ndim < 2:
        P = P.reshape((1,P.size))

    # Allow negative infinity without complaint
    old_settings = np.seterr(divide='ignore')
    logP = np.log2(P)
    np.seterr(**old_settings)

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
    P = P.astype(np.float)
    Q = Q.astype(np.float)

    if P.ndim < 2:
        P = P.reshape((1,P.size))
    if Q.ndim < 2:
        Q = Q.reshape((Q.size,1))

    P_ = np.tile(P.reshape(P.shape[0], P.shape[1], 1), (1,1,Q.shape[1]))
    Q_ = np.tile(Q.reshape(1, Q.shape[0], Q.shape[1]), (P.shape[0],1,1))

    # Allow negative infinity without complaint
    old_settings = np.seterr(divide='ignore')
    logQ_ = np.tile(np.log2(Q.reshape(1, Q.shape[0], Q.shape[1])), 
                    (P.shape[0],1,1))
    np.seterr(**old_settings)

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
    P = P.astype(np.float)
    Q = Q.astype(np.float)

    if P.ndim < 2:
        return np.squeeze(cross_H(P, Q) - H(P))
    out = np.squeeze(cross_H(P, Q) - H(P)[:,np.newaxis])
    if out.ndim == 0:
        return out[()]
    return out


def JS_div(P, Q, metric=False):
    """
    P and Q are arrays of either dimension 1 or 2.

    If P is a matrix, then JS divergence is computed row-wise on P. (P
    is assumed to be left stochastic.) If Q is a matrix, JS divergence
    is computed column-wise on Q. (Q is assumed to be right
    stochastic.)
    
    The square root of the JS divergence is a metric.
    """
    P = P.astype(np.float)
    Q = Q.astype(np.float)

    if P.ndim < 2:
        P = P.reshape((1,P.size))
    if Q.ndim < 2:
        Q = Q.reshape((Q.size,1))
    
    P_ = np.tile(P[:,:,np.newaxis], (1,1,Q.shape[1]))
    Q_ = np.tile(Q[np.newaxis,:,:], (P.shape[0],1,1))
    M_ = .5 * (P_ + Q_)

    # Allow negative infinity without complaint
    old_settings = np.seterr(divide='ignore')
    logM_ = np.log2(M_)
    np.seterr(**old_settings)

    P_zeros = (P_ == 0)
    Q_zeros = (Q_ == 0)
    PQ_zeros = np.bitwise_and(P_zeros, Q_zeros)
    logM_[PQ_zeros] = 0.

    HP = H(P)
    HQ = H(Q.T)

    PM_KLdiv = -1 * (P_ * logM_).sum(1) - HP.reshape(HP.size,1)
    QM_KLdiv = -1 * (Q_ * logM_).sum(1) - HQ.reshape(1,HQ.size) 

    out = .5 * (PM_KLdiv + QM_KLdiv)

    out[out < 0] = 0.
   
    if metric:
        out = np.sqrt(out)

    out = np.squeeze(out)
    if out.ndim == 0:
        return out[()]
    return out


def JS_dist(P, Q):
    """
    P and Q are arrays of either dimension 1 or 2.

    If P is a matrix, then JS distance is computed row-wise on P. (P
    is assumed to be left stochastic.) If Q is a matrix, JS divergence
    is computed column-wise on Q. (Q is assumed to be right
    stochastic.)
    """
    return JS_div(P, Q, metric=True)
