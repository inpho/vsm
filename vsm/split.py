"""
Functions for splitting lists and arrays
"""


import numpy as np


__all__ = ['split_corpus', 'mp_split_ls', 'split_documents']



def split_corpus(arr, indices):
    """
    Splits the given array by the indices into list of sub-arrays.
    
    :param arr: An array to be split.
    :type arr: array
    :param indices: 1-dimensional array of integers that indicates 
        where the array is split.
    :type indices: array
   
    :returns: A list of sub-arrays split at the indices.
   
    **Examples**

    >>> arr = np.arange(8)
    >>> indices = np.array([2,4,7])
    >>> split_corpus(arr, indices)
    [array([0,1]), array([2,3]), array([4,5,6]), array([7])]
    """
    if len(indices) == 0:
        return arr

    if isinstance(indices, list):
        indices = np.array(indices)

    out = np.split(arr, indices)
    
    if (indices >= len(arr)).any():
        out = out[:-1]
    try:
        for i in xrange(len(out)):
            if out[i].size == 0:
                out[i] = np.array([], dtype=arr.dtype)
    except AttributeError:
        for i in xrange(len(out)):
            if out[i].size == 0:
                out[i] = np.array([])

    return out



def mp_split_ls(ls, n):
    """
    Split list into an `n`-length list of arrays.

    :param ls: List to be split.
    :type ls: list

    :param n: Number of splits.
    :type n: int

    :returns: List of arrays whose length is 'n'.

    **Examples**
    >>> ls = [1,5,6,8,2,8]
    >>> mp_split_ls(ls, 4)
    [array([1, 5]), array([6, 8]), array([2]), array([8])]
    """
    return np.array_split(ls, min(len(ls), n))


def split_documents(corpus, indices, n_partitions):
    """
    """
    docs = [(0, indices[0])]
    for i in xrange(len(indices)-1):
        docs.append((indices[i], indices[i+1]))
    docs = np.array(docs, dtype='i8, i8')

    corpus_chunks = np.array_split(corpus, n_partitions)
    chunk_indices = np.cumsum([len(chunk) for chunk in corpus_chunks])
    doc_indices = np.searchsorted(indices, chunk_indices, side='right')
    doc_partitions = np.split(docs, doc_indices[:-1])

    return doc_partitions
