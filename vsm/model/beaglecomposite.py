import numpy as np

from vsm.linalg import row_normalize as _row_normalize
from vsm.model import BaseModel
from vsm.model.beaglecontext import realign_env_mat as _realign_env_mat


class BeagleComposite(BaseModel):
    """
    
    :param ctx_corp: Beagle context corpus object.
    :type ctx_corp: Corpus object

    :param ctx_matrix: Matrix
    :type ctx_matrix: np.ndarray matrix

    :param ord_corp: Beagle order corpus object.
    :type ord_corp: Corpus object

    :param ord_matrix: Matrix
    :type ord_matrix: np.ndarray matrix

    :param context_type: Name of tokenization stored in `corpus` whose
        tokens will be treated as documents. Default is `sentence`.
    :type context_type: string, optional

    :Attributes:
        * **ctx_matrix** (2-D array)
           Matrix that stores normalized beagle context matrix. 
        * **ord_matrix** (2-D array)
           Matrix that stores normalized beagle order matrix. 
        * **context_type** (string)
            Name of tokenization to be treated as documents.

    :Methods:
        * :doc:`beaglecomposite_train`
            Takes an optional argument `wgt` which is `.5` by default.

    :See Also: :class:`vsm.model.BaseModel`.
    """
    def __init__(self, ctx_corp, ctx_matrix, 
                 ord_corp, ord_matrix, context_type='sentence'):
        """
        Assume that the context corpus is a subcorpus of the order
        corpus and that the eventual composite corpus is the context
        corpus. The order matrix is sliced and reordered so that it
        aligns with the context matrix.
        """
        self.ctx_matrix = _row_normalize(ctx_matrix)
        ord_matrix = _realign_env_mat(ctx_corp, ord_corp, ord_matrix)
        self.ord_matrix = _row_normalize(ord_matrix)
        self.context_type = context_type


    def train(self, wgt=.5):
        """
        `wgt` should be a value in [0,1].
        """
        print 'Summing context and order vectors'        
        self.matrix = wgt * self.ctx_matrix + (1 - wgt) * self.ord_matrix


