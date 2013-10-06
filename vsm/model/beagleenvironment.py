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


