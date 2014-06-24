import numpy as np

from base import BaseModel


__all__ = ['BeagleEnvironment']


class BeagleEnvironment(BaseModel):
    """
    `BeagleEnvironment` is a randomly generated fixed vectors
    representing the environment.

    :param corpus: Source of observed data.
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
            Randomly generated environment matrix.             

    :Methods:
        * :doc:`be_train`
            Sets the environment matrix to randomly generated then 
            normalized vectors.
            
    :See Also: :class:`vsm.model.base.BaseModel`
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
        Sets a m x n environment matrix where m is the number of words in
        `corpus` and n is `n_cols`. The matrix consists of randomly generated
        vectors. 
        """
        self.matrix = np.array(np.random.normal(size=self.shape),
                               dtype=self.dtype)
        # normalize rows
        self.matrix /= np.sqrt((self.matrix * self.matrix).sum(1)[:,np.newaxis])


