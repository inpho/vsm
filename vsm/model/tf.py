import multiprocessing as mp

import numpy as np
from scipy.sparse import hstack

from vsm.spatial import count_matrix
from vsm.split import *
from base import *
from mparray import *


__all__ = ['TfSeq', 'TfMulti']



class TfSeq(BaseModel):
    """
    Trains a term-frequency model. 

    In a term-frequency model, the number of occurrences of a word
    type in a context is counted for all word types and contexts. Word
    types correspond to matrix rows and contexts correspond to matrix
    columns.

    :See Also: :class:`vsm.model.base`, :class:`vsm.corpus.Corpus`,
        :class:`scipy.sparse.coo_matrix`
    """
    
    def __init__(self, corpus=None, context_type=None):
        """
        Initialize TfSeq.

        :param corpus: A Corpus object containing the training data.
        :type corpus: Corpus
    
        :param context_type: A string specifying the type of context over which
            the model trainer is applied.
        :type context_type: string
        """

        self.context_type = context_type
        if corpus:
            self.corpus = corpus.corpus
            self.contexts = corpus.view_contexts(context_type, as_slices=True)
            self.m_words = corpus.words.size
        else:
            self.corpus = []
            self.contexts = []
            self.m_words = 0


    def train(self):
        """
        Counts word-type occurrences per context and stores the results in
        `self.matrix`.
        """
        self.matrix = count_matrix(self.corpus, self.contexts, self.m_words)


    
class TfMulti(BaseModel):
    """
    Trains a term-frequency model. 

    In a term-frequency model, the number of occurrences of a word
    type in a context is counted for all word types and contexts. Word
    types correspond to matrix rows and contexts correspond to matrix
    columns.

    The data structure is a sparse integer matrix.

    :See Also: :class:`vsm.model.base.BaseModel`, :class:`vsm.corpus.Corpus`,
        :class:`scipy.sparse.coo_matrix`
    """
    
    def __init__(self, corpus=None, context_type=None):
        """
        Initialize TfMulti.

        :param corpus: A Corpus object containing the training data
        :type corpus: Corpus, optional
    
        :param context_type: A string specifying the type of context over which
            the model trainer is applied.
        :type context_type: string, optional
        """

        self.context_type = context_type
        if corpus:
            self.contexts = corpus.view_contexts(context_type, as_slices=True)
            self._set_corpus(corpus.corpus)
            self._set_m_words(corpus.words.size)
        else:
            self.contexts = []
            self._set_corpus(np.array([], dtype=np.int))
            self._set_m_words(0)


    @staticmethod
    def _set_corpus(arr):
        global _corpus
        _corpus = mp_shared_array(arr)


    @staticmethod
    def _set_m_words(n):
        global _m_words
        _m_words = mp_shared_value(n)

        
    def train(self, n_procs):
        """
        Takes a number of processes `n_procs` over which to map and reduce.

        :param n_procs: Number of processors.
        :type n_procs: int
        """
        ctx_ls = mp_split_ls(self.contexts, n_procs)

        print 'Mapping'
        p=mp.Pool(n_procs)
        # cnt_mats = map(tf_fn, ctx_ls) # For debugging        
        cnt_mats = p.map(tf_fn, ctx_ls)
        p.close()

        print 'Reducing'
        # Horizontally stack TF matrices and store the result
        self.matrix = hstack(cnt_mats, format='coo')


def tf_fn(ctx_sbls):
    """
    The map function for vsm.model.TfMulti. Takes a list of contexts
    as slices and returns a count matrix.
    
    :param ctx_sbls: list of contexts as slices.
    :type ctx_sbls: list of slices

    :returns: a count matrix
    """
    offset = ctx_sbls[0].start
    corpus = _corpus[offset: ctx_sbls[-1].stop]
    slices = [slice(s.start-offset, s.stop-offset) for s in ctx_sbls]
    return count_matrix(corpus, slices, _m_words.value)
