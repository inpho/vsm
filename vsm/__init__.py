import multiprocessing as mp
import numpy as np


def arr_add_field(arr, new_field, vals):

    # Constructing new dtype
    new_dtype = np.array(vals).dtype
    dt = [(n, arr.dtype[n]) for n in arr.dtype.names]
    dt.append((new_field, new_dtype))

    # Building new structured array
    new_arr = np.empty_like(arr, dtype=dt)
    for n in new_arr.dtype.names:
        if n == new_field:
            new_arr[n][:] = vals[:]
        else:
            new_arr[n][:] = arr[n][:]

    return new_arr
    

def enum_matrix(arr, axis=0, indices=[], field_name='i'):
    """
    Takes a 1-dimensional or 2-dimensional array and returns a sorted
    structured array with indices.
    """
    if len(indices) == 0:
        indices = np.arange(arr.shape[1])
    ind = np.array([indices.copy() for i in xrange(arr.shape[0])])
    dt = [(field_name, indices.dtype), ('value', arr.dtype)]
    mt = zip_arr(ind, arr, field_names=[field_name, 'value'])

    if len(arr.shape) > 1:  
	if axis:
	    for i in xrange(arr.shape[axis]):
	        idx = np.argsort(mt['value'][:,i])
	        mt[field_name][:,i] = ind[:,i][idx]
                mt['value'][:,i] = arr[:,i][idx]
	        mt[:,i] = mt[:,i][::-1]	
	else: 
            for i in xrange(arr.shape[axis]):
	        idx = np.argsort(mt['value'][i])
	        mt[field_name][i] = ind[i][idx]
                mt['value'][i] = arr[i][idx]
	        mt[i,:] = mt[i,:][::-1]

    else:
	idx = np.argsort(arr)
	mt[field_name][:] = idx[:]
	mt['value'][:] = arr[idx]

    return mt


    
def enum_sort(arr, indices=[], field_name='i', filter_nan=False):
    """
    Takes a 1-dimensional array and returns a sorted structured array.
    """
    idx = np.argsort(arr)
    if len(indices) == 0:
	indices = np.arange(arr.shape[0])
    else:
	indices = np.array(indices)
	
    dt = [(field_name, indices.dtype), ('value', arr.dtype)]

    new_arr = np.empty(shape=arr.shape, dtype=dt)
    new_arr[field_name] = indices[idx]
    new_arr['value'] = arr[idx]

    if filter_nan:
        new_arr = new_arr[np.isfinite(new_arr['value'])]
        
    return new_arr[::-1]



def enum_array(a, indices=None, field_name='i'):
    """
    """
    a1 = np.arange(a.size)

    if indices == None:
    	return zip_arr(a1, a, field_names=[field_name, 'value'])    
    else:
	return zip_arr(indices, a, field_names=[field_name, 'value'])



def zip_arr(arr_1, arr_2, field_names=['arr_1','arr_2']):
    """
    Takes two arrays and returns a zipped structured array.
    """
    if field_names:
        dt = [(field_names[0], arr_1.dtype), (field_names[1], arr_2.dtype)]

    new_arr = np.empty_like(arr_1, dtype=dt)

    new_arr[new_arr.dtype.names[0]][:] = arr_1[:]
    new_arr[new_arr.dtype.names[1]][:] = arr_2[:]
    
    return new_arr


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


def mp_split_ls(ls, n):
    """
    Split list into an `n`-length list of arrays
    """
    return np.array_split(ls, min(len(ls), n))


def mp_shared_array(arr, ctype='i'):
    
    shared_arr_base = mp.Array(ctype, arr.size)
    shared_arr_base[:] = arr
    return np.ctypeslib.as_array(shared_arr_base.get_obj())


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


def arr_add_field_test():

    arr = np.array([(1, '1'), (2, '2'), (3, '3')],
                   dtype=[('i', np.int), ('c', '|S1')])
    new_arr = np.array([(1, '1', 0), (2, '2', 0), (3, '3', 0)],
                       dtype=[('i', np.int), ('c', '|S1'), ('new', np.int)])

    new_field = 'new'
    vals = np.zeros(3, dtype=np.int)

    test_arr = arr_add_field(arr, new_field, vals)

    assert (new_arr==test_arr).all()
    assert new_arr.dtype==test_arr.dtype



def test_map_strarr():

    arr = np.array([(0, 1.), (1, 2.)], 
                   dtype=[('i', 'i4'), ('v', 'f4')])
    m = ['foo', 'bar']
    arr = map_strarr(arr, m, 'i', new_k='str')

    assert (arr['str'] == np.array(m, dtype=np.array(m).dtype)).all()
    assert (arr['v'] == np.array([1., 2.], dtype='f4')).all()


def mp_split_ls_test():

    l = [slice(0,0), slice(0,0), slice(0,0)]
    assert len(mp_split_ls(l, 1)) == 1
    assert (mp_split_ls(l, 1)[0] == l).all()
    assert len(mp_split_ls(l, 2)) == 2
    assert (mp_split_ls(l, 2)[0] == [slice(0,0), slice(0,0)]).all()
    assert (mp_split_ls(l, 2)[1] == [slice(0,0)]).all()
    assert len(mp_split_ls(l, 3)) == 3
    assert (mp_split_ls(l, 3)[0] == [slice(0,0)]).all()
    assert (mp_split_ls(l, 3)[1] == [slice(0,0)]).all()
    assert (mp_split_ls(l, 3)[2] == [slice(0,0)]).all()
