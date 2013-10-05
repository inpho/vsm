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
        self.matrix = np.array(np.random.normal(size=self.shape),
                               dtype=self.dtype)
        self.matrix = row_normalize(self.matrix)


