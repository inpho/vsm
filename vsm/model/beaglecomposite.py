import numpy as np

from base import BaseModel
from beaglecontext import realign_env_mat as _realign_env_mat


__all__ = ['BeagleComposite']


class BeagleComposite(BaseModel):
    """
    `BeagleComposite` combines the BEAGLE order and context model
    with a user defined ratio. Default ratio is .5 which weighs
    order and context matrices equally.

    :param ctx_corp: Corpus from BEAGLE context model.
    :type ctx_corp: Corpus object

    :param ctx_matrix: BEAGLE context matrix.
    :type ctx_matrix: np.ndarray matrix

    :param ord_corp: Corpus from BEAGLE order model.
    :type ord_corp: Corpus object

    :param ord_matrix: BEAGLE order matrix.
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
        * **matrix** (2-D array)
            
    :Methods:
        * :doc:`beaglecomposite_train`
            Combines context and order model with user-defined `wgt`.

    :See Also: :class:`vsm.model.base.BaseModel`.
    """
    def __init__(self, ctx_corp, ctx_matrix, 
                 ord_corp, ord_matrix, context_type='sentence'):
        """
        Assume that the context corpus is a subcorpus of the order
        corpus and that the eventual composite corpus is the context
        corpus. The order matrix is sliced and reordered so that it
        aligns with the context matrix.
        """
        self.ctx_matrix /= np.sqrt((ctx_matrix * ctx_matrix).sum(1)[:,np.newaxis])
        ord_matrix = _realign_env_mat(ctx_corp, ord_corp, ord_matrix)
        self.ord_matrix /= np.sqrt((ord_matrix * ord_matrix).sum(1)[:,np.newaxis])
        self.context_type = context_type


    def train(self, wgt=.5):
        """
        :param wgt: The weight of context model. If `wgt` is .7 then
            the ratio of context and order model is 7:3. `wgt` should be 
            a value in [0,1]. Default is .5.
        :type wgt: float, optional
        """
        print 'Summing context and order vectors'        
        self.matrix = wgt * self.ctx_matrix + (1 - wgt) * self.ord_matrix

