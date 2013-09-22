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
    
    if type(indices) == list:
        indices = np.array(indices)
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


def mp_shared_value(x, ctype='i'):

    return mp.Value(ctype, x)


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

