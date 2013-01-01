import numpy as np
from scipy import sparse



def enum_array(a):

    a = np.array(list(enumerate(a)),
                 dtype=[('i', np.int32),('v', a.dtype)])

    return a



def enum_sort(a):

    a = enum_array(a)

    a.sort(order='v')
    
    a = a[::-1]

    return a



def map_strarr(arr, m, k, new_k=None):
    """
    Takes a structured array `arr`, a field name `k` and an Indexable
    `m` and returns a copy of `arr` with its field `k` values mapped
    according to `m`. If `new_name` is given, the field name `k` is
    replaced with `new_name`.
    """
    if not new_k:

        new_k = k

    k_vals = np.array([m[i] for i in arr[k]])

    dt = [(n, arr.dtype[n]) for n in arr.dtype.names]

    i = arr.dtype.names.index(k)

    dt[i] = (new_k, k_vals.dtype)

    new_arr = np.empty_like(arr, dtype=dt)

    for n in new_arr.dtype.names:
        
        if n == new_k:

            new_arr[new_k][:] = k_vals[:]

        else:

            new_arr[n][:] = arr[n][:]

    return new_arr



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

    assert np.allclose(row_norms(m), np.ones(5))



def test_map_strarr():

    arr = np.array([(0, 1.), (1, 2.)], 
                   dtype=[('i', 'i4'), ('v', 'f4')])

    m = ['foo', 'bar']

    arr = map_strarr(arr, m, 'i', new_k='str')

    assert (arr['str'] == np.array(m, dtype=np.array(m).dtype)).all()

    assert (arr['v'] == np.array([1., 2.], dtype='f4')).all()
