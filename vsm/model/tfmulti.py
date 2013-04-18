import multiprocessing as mp

import numpy as np
from scipy import sparse

from vsm.model import BaseModel
from vsm.model.tf import TfModel as TfSeq
from vsm.corpus import Corpus



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


#TODO: Replace this with the subcorpus method when it has a parameter
#for preserving the original word list
def split_corpus(corpus, context_type, n):

    contexts = corpus.view_contexts(context_type, as_slices=True)

    # Split contexts into an `n`-length list of lists of contexts
    if n == 1:
        ctx_ls = [contexts]
    else:
        ctx_ls = np.array_split(contexts, n-1)
        if len(ctx_ls) != n:
            ctx_ls = np.array_split(contexts, n)

    corp_ls = []
    for ctx_sbls in ctx_ls:
        corp_slice = slice(ctx_sbls[0].start, ctx_sbls[-1].stop)
        sbcorp = Corpus([])
        sbcorp.corpus = corpus.corpus[corp_slice]
        cdata = np.array([(s.stop - corp_slice.start,) for s in ctx_sbls], 
                         dtype=[('idx', '<i8')])
        sbcorp.context_data = [cdata]
        sbcorp.context_types = [context_type]
        sbcorp.words = corpus.words
        sbcorp.words_int = corpus.words_int
        corp_ls.append(sbcorp)

    return corp_ls



class TfMulti(BaseModel):
    """
    """
    def __init__(self, corpus, context_type):

        self.corpus = corpus
        self.context_type = context_type


    def train(self, n_procs):

        # Split the corpus into subcorpora to map over
        corp_ls = split_corpus(self.corpus, self.context_type, n_procs)
        corp_ls = [(sbcorp, self.context_type) for sbcorp in corp_ls]

        print 'Mapping'
        p=mp.Pool(n_procs)
        # models = map(tf_fn, corp_ls) # For debugging        
        models = p.map(tf_fn, corp_ls)
        p.close()

        print 'Reducing'
        # Horizontally stack TF matrices and store the result
        self.matrix = hstack_coo([m.matrix for m in models])


def tf_fn((sbcorp, ctx_type)):
    """
    """
    m = TfSeq(sbcorp, ctx_type)
    m.train()
    return m




def hstack_coo_test():

    dense_mat_ls = [np.random.random((3,4)),
                    np.random.random((3,5)),
                    np.random.random((3,6))]

    mat_ls = [sparse.coo_matrix(m) for m in dense_mat_ls]
    
    assert (np.hstack(dense_mat_ls) == hstack_coo(mat_ls).toarray()).all()


def split_corpus_test():
    
    corpus = np.array([0, 3, 2, 1, 0, 3, 0, 2, 3, 0, 2, 3, 1, 2, 0, 3, 
                       2, 1, 2, 2], dtype=np.int)
    context_data = np.array([(3,), (5,), (7,), (11,), (11,), (15,), (18,), 
                             (20,)], dtype=[('idx', '<i8')])
    corpus = Corpus(corpus, context_data=[context_data], 
                    context_types=['document'])

    corp_ls = split_corpus(corpus, 'document', 3)
    assert len(corp_ls) == 3
    assert (corp_ls[0].words == corpus.words).all()
    assert (corp_ls[1].words == corpus.words).all()
    assert (corp_ls[2].words == corpus.words).all()
    contexts0 = corp_ls[0].view_contexts('document')
    for i in xrange(3):
        assert (contexts0[i] == corpus.view_contexts('document')[i]).all()
    contexts1 = corp_ls[1].view_contexts('document')
    for i in xrange(2):
        assert (contexts1[i] == corpus.view_contexts('document')[i+3]).all()
    contexts2 = corp_ls[2].view_contexts('document')
    for i in xrange(2):
        assert (contexts2[i] == corpus.view_contexts('document')[i+5]).all()



def TfMulti_test():

    from vsm.util.corpustools import random_corpus

    c = random_corpus(10000, 100, 0, 100, context_type='document')

    m0 = TfMulti(c, 'document')
    m0.train(n_procs=3)

    m1 = TfSeq(c, 'document')
    m1.train()

    assert (m0.matrix.toarray() == m1.matrix.toarray()).all()

    # I/O
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
