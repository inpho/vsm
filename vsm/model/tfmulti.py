import multiprocessing as mp

import numpy as np
from scipy import sparse

from vsm.model import BaseModel
from vsm.model.tf import TfModel as TfSeq


def count_matrix(arr, slices, m=None):
    """
    arr_ls : list of numpy integer arrays
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

    return sparse.coo_matrix((data, (row_indices, col_indices)),
                             shape=shape, dtype=np.int32)

    
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

    return sparse.coo_matrix((data, (row, col)),
                             shape=(shape_0, shape_1),
                             dtype=data.dtype)


def mp_split_ls(ls, n):

    # Split list into an `n`-length list of arrays
    return np.array_split(ls, min(len(ls), n))


def mp_shared_array(arr, ctype='i'):
    
    shared_arr_base = mp.Array(ctype, arr.size)
    shared_arr_base[:] = arr
    return np.ctypeslib.as_array(shared_arr_base.get_obj())



class TfMulti(BaseModel):
    """
    """
    def __init__(self, corpus, context_type):

        self.context_type = context_type
        self.contexts = corpus.view_contexts(context_type, as_slices=True)
        
        global _corpus
        _corpus = mp_shared_array(corpus.corpus)

        global _m_words
        _m_words = mp.Value('i', corpus.words.size)


    def train(self, n_procs):

        ctx_ls = mp_split_ls(self.contexts, n_procs)

        print 'Mapping'
        p=mp.Pool(n_procs)
        # cnt_mats = map(tf_fn, ctx_ls) # For debugging        
        cnt_mats = p.map(tf_fn, ctx_ls)
        p.close()

        print 'Reducing'
        # Horizontally stack TF matrices and store the result
        self.matrix = hstack_coo(cnt_mats)


def tf_fn(ctx_sbls):
    """
    """
    offset = ctx_sbls[0].start
    corpus = _corpus[offset: ctx_sbls[-1].stop]
    slices = [slice(s.start-offset, s.stop-offset) for s in ctx_sbls]
    return count_matrix(corpus, slices, _m_words.value)




def hstack_coo_test():

    dense_mat_ls = [np.random.random((3,4)),
                    np.random.random((3,5)),
                    np.random.random((3,6))]

    mat_ls = [sparse.coo_matrix(m) for m in dense_mat_ls]
    
    assert (np.hstack(dense_mat_ls) == hstack_coo(mat_ls).toarray()).all()


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


def count_matrix_test():

    arr = [1, 2, 4, 2, 1]
    slices = [slice(0,1), slice(1, 3), slice(3,3), slice(3, 5)]
    m = 6
    result = sparse.coo_matrix([[0, 0, 0, 0],
                                [1, 0, 0, 1],
                                [0, 1, 0, 1],
                                [0, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 0, 0]])

    assert (result.toarray() == count_matrix(arr, slices, m).toarray()).all()


def TfMulti_test():

    from vsm.util.corpustools import random_corpus

    c = random_corpus(1000000, 4000, 0, 100, context_type='document')

    m0 = TfMulti(c, 'document')
    m0.train(n_procs=4)

    m1 = TfSeq(c, 'document')
    m1.train()

    assert (m0.matrix.toarray() == m1.matrix.toarray()).all()

    #I/O
    from tempfile import NamedTemporaryFile
    import os

    try:
        tmp = NamedTemporaryFile(delete=False, suffix='.npz')
        m0.save(tmp.name)
        tmp.close()
        m1 = TfMulti.load(tmp.name)
        assert (m0.matrix.todense() == m1.matrix.todense()).all()
    
    finally:
        os.remove(tmp.name)
