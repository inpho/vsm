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
        self.corpus = corpus
        self.context_type = context_type


    def train(self):
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


    def combine_models(m1, m2):
        """Takes two models. Chooses which model is bigger and clones it to a third
        model. It then iterates over the second model adding values that already
        exists into the third model and creating new entries for those that
        didn't already exist"""
        size_1 = m1.get_shape()
        size_2 = m2.get_shape()
        data_1 = m1.data
        data_2 = m2.data
        size_of_data3 = 0
    
        if np.size(data_1) >= np.size(data_2):
            size_of_data3 = np.size(data_1)
            while np.size(data_1) != np.size(data_2):
                np.append(data_2, 0)
        else:
            size_of_data3 = np.size(data_2)
            while np.size(data_2) != np.size(data_1):
                np.append(data_1, 0)
            
        words_in_data1 = 0
        data_3 = np.concatenate(data_1, data_2)
        data_3 = np.trim_zeros(data_3)
        rows = np.arrange(np.size(data_3))
        cols = np.arrange(np.size(data_3))
        coo_in = (data_3, (rows, cols))
        self.model_3 = sparse.coo_matrix(coo_in, shape=((np.size(data_3)), (np.size(data_3))))
        return model_3
    
    
    def combine_corpus(c1, c2):
        """Takes two corpora and combines them into one super corpus. Also keeps an association list
        between the two old corpora and the new corpus."""
        
        from vsm.corpus import Corpus
        self.c1dict = {}
        self.c2dict = {}
        text = [c1.words[i] for i in c1.corpus] + [c2.words[i] for i in c2.corpus]
        self.c3 = Corpus(text)
        for i,word in enumerate(c3.words):
            if word in c1.words:
                self.c1dict[word] = i
            if word in c2.words:
                self.c2dict[word] = i

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
