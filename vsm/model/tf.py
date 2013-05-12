import multiprocessing as mp
import numpy as np

from vsm import mp_split_ls, mp_shared_array
from vsm.model import BaseModel
from vsm.linalg import hstack_coo, count_matrix


class TfSeq(BaseModel):
    """
    Trains a term-frequency model. 

    In a term-frequency model, the number of occurrences of a word
    type in a context is counted for all word types and contexts. Word
    types correspond to matrix rows and contexts correspond to matrix
    columns.

    The data structure is a sparse integer matrix.

    Parameters
    ----------
    corpus : Corpus
        A Corpus object containing the training data
    context_type : string
        A string specifying the type of context over which the model
        trainer is applied.

    Attributes
    ----------
    corpus : Corpus
        A Corpus object containing the training data
    context_type : string
        A string specifying the type of context over which the model
        trainer is applied.
    matrix : scipy.sparse.coo_matrix
        A sparse matrix in 'coordinate' format that contains the
        frequency counts.

    Methods
    -------
    train
        Counts word-type occurrences per context and stores the
        results in `self.matrix`
    save
        Takes a filename or file object and saves `self.matrix` and
        `self.context_type` in an npz archive.
    load
        Takes a filename or file object and loads it as an npz archive
        into a BaseModel object.

    See Also
    --------
    BaseModel
    vsm.corpus.Corpus
    scipy.sparse.coo_matrix
    """
    def __init__(self, corpus, context_type):

        self.context_type = context_type
        self.corpus = corpus.corpus
        self.contexts = corpus.view_contexts(context_type, as_slices=True)
        self.m_words = corpus.words.size


    def train(self):
        self.matrix = count_matrix(self.corpus, self.contexts, self.m_words)




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




def TfMulti_test():

    from vsm.util.corpustools import random_corpus

    c = random_corpus(1000000, 10000, 0, 1000, context_type='document')

    m0 = TfMulti(c, 'document')
    m0.train(n_procs=4)

    m1 = TfSeq(c, 'document')
    m1.train()

    # assert (m0.matrix.toarray() == m1.matrix.toarray()).all()

    #I/O
    # from tempfile import NamedTemporaryFile
    # import os

    # try:
    #     tmp = NamedTemporaryFile(delete=False, suffix='.npz')
    #     m0.save(tmp.name)
    #     tmp.close()
    #     m1 = TfMulti.load(tmp.name)
    #     assert (m0.matrix.todense() == m1.matrix.todense()).all()
    
    # finally:
    #     os.remove(tmp.name)
