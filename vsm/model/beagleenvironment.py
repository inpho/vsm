import numpy as np

from vsm.linalg import row_normalize
from vsm.model import BaseModel


class BeagleEnvironment(BaseModel):

    def __init__(self, corpus, n_cols=2048, dtype=np.float64, 
                 context_type='sentence'):
        """
        """
        self.context_type = context_type
        self.shape = (corpus.words.shape[0], n_cols)
        self.dtype = dtype


    def train(self):
        """
        """
        self.matrix = np.array(np.random.random(self.shape), 
                               dtype=self.dtype)
        self.matrix = self.matrix * 2 - 1
        self.matrix = row_normalize(self.matrix)



def test_BeagleEnvironment():

    from vsm.util.corpustools import random_corpus
    from vsm.linalg import row_norms

    c = random_corpus(1000, 100, 0, 20)

    m = BeagleEnvironment(c, n_cols=100)
    m.train()

    assert (m.matrix <= 1).all()
    assert (m.matrix >= -1).all()

    norms = row_norms(m.matrix)

    assert np.allclose(np.ones(norms.shape[0]), norms)

    from tempfile import NamedTemporaryFile
    import os

    try:
        tmp = NamedTemporaryFile(delete=False, suffix='.npz')
        m.save(tmp.name)
        tmp.close()
        m1 = BeagleEnvironment.load(tmp.name)
        assert (m.matrix == m1.matrix).all()
    
    finally:
        os.remove(tmp.name)
