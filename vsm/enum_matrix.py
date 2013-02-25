import numpy as np
from vsm import enum_sort

def enum_matrix(arr, axis=0, indices=None, field_name='i'):

    if not indices:
        indices = np.arange(arr.shape[0])
    dt = [(field_name, indices.dtype), ('value', arr.dtype)]
    mt = np.empty(shape=arr.shape, dtype=dt)

    mt['value'][:, :] = arr[:, :]

    for i in xrange(arr.shape[1]):
        mt[field_name][:, i] = indices[:]

    for i in xrange(mt.shape[1]):
        mt[:, i].sort(order='value')
        mt[:, i] = mt[:, i][::-1]

    return mt


def test_enum_matrix():

    return enum_matrix(arr)


if __name__=='__main__':

    arr = np.random.random((200000, 60))
