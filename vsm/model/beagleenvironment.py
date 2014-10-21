import numpy as np

from base import BaseModel


__all__ = ['BeagleEnvironment']


class BeagleEnvironment(BaseModel):
    """
    `BeagleEnvironment` is a randomly generated fixed vectors
    representing the environment.
    """
    
    def __init__(self, corpus, n_cols=2048, dtype=np.float64, 
                 context_type='sentence'):
        """
        Initialize BeagleEnvironment.

        :param corpus: Source of observed data.
        :type corpus: Corpus

        :param n_cols: Number of columns. Default is 2048.
        :type n_cols: int, optional

        :param dtype: Numpy dtype for matrix attribute. Default is `np.float64`.
        :type dtype: np.dtype, optional

        :param context_type: Name of tokenization stored in `corpus` whose
            tokens will be treated as documents. Default is `sentence`.
        :type context_type: string, optional
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


