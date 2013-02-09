import numpy as np



def enum_array(a):
    """
    """
    return np.array(list(enumerate(a)),
                    dtype=[('i', np.int32),('value', a.dtype)])


def enum_sort(a, filter_nan=False):
    """
    """
    a = enum_array(a)
    a.sort(order='value')
    a = a[::-1]

    if filter_nan:
        a = a[np.isfinite(a['value'])]
        
    return a


def map_strarr(arr, m, k, new_k=None):
    """
    Takes a structured array `arr`, a field name `k` and an Indexable
    `m` and returns a copy of `arr` with its field `k` values mapped
    according to `m`. If `new_name` is given, the field name `k` is
    replaced with `new_name`.
    """
    # Constructing new dtype
    if not new_k:
        new_k = k
    k_vals = np.array([m[i] for i in arr[k]])
    dt = [(n, arr.dtype[n]) for n in arr.dtype.names]
    i = arr.dtype.names.index(k)
    dt[i] = (new_k, k_vals.dtype)

    # Building new structured array
    new_arr = np.empty_like(arr, dtype=dt)
    for n in new_arr.dtype.names:
        if n == new_k:
            new_arr[new_k][:] = k_vals[:]
        else:
            new_arr[n][:] = arr[n][:]

    return new_arr



def isstr(x):
    """
    """
    return isinstance(x, basestring) or isinstance(x, np.flexible)



def isint(x):
    """
    """
    return (isinstance(x, np.integer) 
            or isinstance(x, int) or isinstance(x, long))



def isfloat(x):
    """
    """
    return (isinstance(x, np.inexact) or isinstance(x, np.float))



#
# Testing
#



def test_map_strarr():

    arr = np.array([(0, 1.), (1, 2.)], 
                   dtype=[('i', 'i4'), ('v', 'f4')])
    m = ['foo', 'bar']
    arr = map_strarr(arr, m, 'i', new_k='str')

    assert (arr['str'] == np.array(m, dtype=np.array(m).dtype)).all()
    assert (arr['v'] == np.array([1., 2.], dtype='f4')).all()
