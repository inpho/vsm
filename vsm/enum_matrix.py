import numpy as np
from vsm import enum_sort

def enum_matrix(arr, axis=0, indices=[], field_name='i'):

    if len(indices) == 0:
        indices = np.arange(arr.shape[0])
    dt = [(field_name, indices.dtype), ('value', arr.dtype)]
    mt = np.empty(shape=arr.shape, dtype=dt)

    print 'Assigning values'

    mt['value'][:, :] = arr[:, :]

    print 'Applying indices'

    for i in xrange(arr.shape[1]):
        mt[field_name][:, i] = indices[:]

    print 'Sorting'

    for i in xrange(mt.shape[1]):
        mt[:, i].sort(order='value')
        
    print 'Reversing'
        
    for i in xrange(mt.shape[1]):
        mt[:, i] = mt[:, i][::-1]

    return mt


def test_enum_matrix():

    return enum_matrix(arr, indices=indices)


if __name__=='__main__':

    n = 200000
    indices = np.array([str(i) for i in xrange(n)])
    arr = np.random.random((n, 60))
