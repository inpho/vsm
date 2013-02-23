import numpy as np
from scipy import sparse

from vsm.model import BaseModel


class TfModel(BaseModel):
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
        """
        """
        self.corpus = corpus
        self.context_type = context_type


    def train(self):
        """
        """
        docs = self.corpus.view_contexts(self.context_type)
        shape = (self.corpus.words.size, len(docs))

        print 'Computing term frequencies'
        data = np.ones_like(self.corpus.corpus)
        row_indices = self.corpus.corpus
        col_indices = np.empty_like(self.corpus.corpus)
        j, k = 0, 0

        for i,token in enumerate(docs):
            k += len(token)
            col_indices[j:k] = i
            j = k

        coo_in = (data, (row_indices, col_indices))
        self.matrix = sparse.coo_matrix(coo_in, shape=shape, 
                                        dtype=np.int32)



def test_TfModel():

    from vsm.util.corpustools import random_corpus

    c = random_corpus(10000, 100, 0, 100, context_type='document')

    m = TfModel(c, 'document')
    m.train()

    from tempfile import NamedTemporaryFile
    import os

    try:
        tmp = NamedTemporaryFile(delete=False, suffix='.npz')
        m.save(tmp.name)
        tmp.close()
        m1 = TfModel.load(tmp.name)
        assert (m.matrix.todense() == m1.matrix.todense()).all()
    
    finally:
        os.remove(tmp.name)

    return m.matrix
