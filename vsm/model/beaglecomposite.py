import numpy as np

from base import BaseModel
from beaglecontext import realign_env_mat


__all__ = [ 'BeagleComposite' ]


class BeagleComposite(BaseModel):
    """
    `BeagleComposite` combines the BEAGLE order and context model
    with a user defined ratio. Default ratio is .5 which weighs
    order and context matrices equally.
    """
    
    def __init__(self, ctx_corp, ctx_matrix, 
                 ord_corp, ord_matrix, context_type='sentence'):
        """
        Assume that the context corpus is a subcorpus of the order
        corpus and that the eventual composite corpus is the context
        corpus. The order matrix is sliced and reordered so that it
        aligns with the context matrix.
 
        :param ctx_corp: Corpus from BEAGLE context model.
        :type ctx_corp: :class:`Corpus`
    
        :param ctx_matrix: BEAGLE context matrix.
        :type ctx_matrix: np.ndarray matrix

        :param ord_corp: Corpus from BEAGLE order model.
        :type ord_corp: :class:`Corpus`

        :param ord_matrix: BEAGLE order matrix.
        :type ord_matrix: np.ndarray matrix

        :param context_type: Name of tokenization stored in `corpus` whose
            tokens will be treated as documents. Default is `sentence`.
        :type context_type: string, optional
        """
        self.ctx_matrix = (ctx_matrix / 
                           ((ctx_matrix**2).sum(1)**0.5)[:,np.newaxis])
        self.ord_matrix = realign_env_mat(ctx_corp, ord_corp, ord_matrix)
        self.ord_matrix /= ((self.ord_matrix**2).sum(1)**0.5)[:,np.newaxis]
        self.context_type = context_type


    def train(self, wgt=.5):
        """
        Combines the context and order matrices blended by `wgt` ratio.

        :param wgt: The weight of context model. If `wgt` is .7 then
            the ratio of context and order model is 7:3. `wgt` should be 
            a value in [0,1]. Default is .5.
        :type wgt: float, optional
       
        :returns: `None`
        """
        print 'Summing context and order vectors'        
        self.matrix = wgt * self.ctx_matrix + (1 - wgt) * self.ord_matrix

