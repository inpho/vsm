import numpy as np
from scipy import sparse

from vsm.model import BaseModel


class TfModel(BaseModel):
    """
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
