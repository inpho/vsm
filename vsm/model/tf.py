import multiprocessing as mp
import platform, warnings

import numpy as np
from scipy.sparse import hstack

from vsm.spatial import count_matrix
from vsm.split import *
from base import *


__all__ = ['TF', 'TfSeq', 'TfMulti']



class TfSeq(BaseModel):
    """
    Trains a term-frequency model. 

    In a term-frequency model, the number of occurrences of a word
    type in a context is counted for all word types and documents. Word
    types correspond to matrix rows and documents correspond to matrix
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
            self.docs = corpus.view_contexts(context_type, as_slices=True)
            self.V = corpus.words.size
        else:
            self.corpus = []
            self.docs = []
            self.V = 0


    def train(self):
        """
        Counts word-type occurrences per context and stores the results in
        `self.matrix`.
        """
        self.matrix = count_matrix(self.corpus, self.docs, self.V)


    
class TfMulti(TfSeq):
    """
    Trains a term-frequency model. 

    In a term-frequency model, the number of occurrences of a word
    type in a context is counted for all word types and documents. Word
    types correspond to matrix rows and documents correspond to matrix
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
        self._read_globals = False
        self._write_globals = False

        super(TfMulti, self).__init__(corpus=corpus, context_type=context_type)


    def _move_globals_to_locals(self):
        
        self._write_globals = False
        self.V = self.V
        self.corpus = self.corpus
        self._read_globals = False
        global _V, _corpus
        del _V, _corpus


    def _move_locals_to_globals(self):
        
        self._write_globals = True
        self.V = self.V
        self.corpus = self.corpus
        self._read_globals = True
        del self._V_local, self._corpus_local


    @property
    def corpus(self):
        if self._read_globals:
            return np.frombuffer(_corpus, np.int32)
        return self._corpus_local

    @corpus.setter
    def corpus(self, a):
        if self._write_globals:
            global _corpus
            if not '_corpus' in globals():
                _corpus = mp.Array('i', len(a), lock=False)
            _corpus[:] = a
        else:
            self._corpus_local = a

    @property
    def V(self):
        if self._read_globals:
            return _V.value
        return self._V_local

    @V.setter
    def V(self, V):
        if self._write_globals:
            global _V
            _V = mp.Value('i', V, lock=False)
        else:
            self._V_local = V



    def train(self, n_proc=2):
        """
        Takes a number of processes `n_proc` over which to map and reduce.

        :param n_procs: Number of processors.
        :type n_procs: int
        """
        self._move_locals_to_globals()

        doc_indices = mp_split_ls(self.docs, n_proc)

        p=mp.Pool(n_proc)
        cnt_mats = p.map(tf_fn, doc_indices)
        p.close()

        self.matrix = hstack(cnt_mats, format='coo')

        self._move_globals_to_locals()



def tf_fn(ctx_sbls):
    """
    The map function for vsm.model.TfMulti. Takes a list of documents
    as slices and returns a count matrix.
    
    :param ctx_sbls: list of documents as slices.
    :type ctx_sbls: list of slices

    :returns: a count matrix
    """
    offset = ctx_sbls[0].start
    corpus = _corpus[offset: ctx_sbls[-1].stop]
    slices = [slice(s.start-offset, s.stop-offset) for s in ctx_sbls]
    return count_matrix(corpus, slices, _V.value)


class TF(object):
    """
    Depending on the boolean parameter `multiprocessing`, returns and
    initializes an instance of either TfSeq or TfMulti.

    Note that on Windows platforms, `multiprocessing` is not implemented.
    In contrast to LdaCgsMulti, LDA always returns a valid object. Instead
    of raising a NotImplementedError, LDA issues a RuntimeWarning, notifying 
    the user the sequental algorithm is being used.
    """
    def __new__(cls, corpus=None, context_type=None, multiprocessing=False):

        kwargs = dict(corpus=corpus, context_type=context_type)
        
        if multiprocessing and platform.system() != 'Windows':
            return TfMulti(**kwargs)
        else:
            if platform.system() == 'Windows':
                warnings.warn("""Multiprocessing is not implemented on Windows.
                Defaulting to sequential algorithm.""", RuntimeWarning)
            return TfSeq(**kwargs)
