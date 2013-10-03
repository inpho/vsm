import numpy as np

from vsm.linalg import row_normalize
from vsm.model import BaseModel


class BeagleEnvironment(BaseModel):
    """

    :param corpus: 
    :type corpus: Corpus object

    :param n_cols: Number of columns. Default is 2048.
    :type n_cols: int, optional

    :param dtype: Numpy dtype for matrix attribute. Default is `np.float64`.
    :type dtype: np.dtype, optional

    :param context_type: Name of tokenization stored in `corpus` whose
        tokens will be treated as documents. Default is `sentence`.
    :type context_type: string, optional

    :Attributes:
        * **context_type** (string)
            Name of tokenization whose tokens will be treated as documents.
        * **shape** (tuple)
            Shape for the matrix.
        * **dtype** (np.dtype)
            Dtype for the matrix.
        * **matrix** (2-D array)
            ?

    :Methods:
        * :doc:`be_train`
            Trains the model.

    :See Also: :class:`vsm.model.BaseModel`
    """
    def __init__(self, corpus, n_cols=2048, dtype=np.float64, 
                 context_type='sentence'):
        """
        """
        self.context_type = context_type
        self.shape = (corpus.words.shape[0], n_cols)
        self.dtype = dtype


    def train(self):
        """
        Trains the model.
        """
        self.matrix = np.array(np.random.normal(size=self.shape),
                               dtype=self.dtype)
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
