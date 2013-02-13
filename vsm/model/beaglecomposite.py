import numpy as np

from vsm.linalg import row_normalize as _row_normalize
from vsm.model import BaseModel
from vsm.model.beaglecontext import realign_env_mat as _realign_env_mat



class BeagleComposite(BaseModel):

    def __init__(self, ctx_corp, ctx_matrix, 
                 ord_corp, ord_matrix, tok_name='sentence'):
        """
        Assume that the context corpus is a subcorpus of the order
        corpus and that the eventual composite corpus is the context
        corpus. The order matrix is sliced and reordered so that it
        aligns with the context matrix.
        """
        self.ctx_matrix = _row_normalize(ctx_matrix)
        ord_matrix = _realign_env_mat(ctx_corp, ord_corp, ord_matrix)
        self.ord_matrix = _row_normalize(ord_matrix)
        self.tok_name = tok_name


    def train(self, wgt=.5):
        """
        `wgt` should be a value in [0,1].
        """
        print 'Summing context and order vectors'        
        self.matrix = wgt * self.ctx_matrix + (1 - wgt) * self.ord_matrix



#
# for testing
#

def test_BeagleComposite():

    from vsm.util.corpustools import random_corpus
    from vsm.model.beagleenvironment import BeagleEnvironment
    from vsm.model.beaglecontext import BeagleContextSeq
    from vsm.model.beagleorder import BeagleOrderSeq

    ec = random_corpus(1000, 50, 0, 20, tok_name='sentence')
    cc = ec.apply_stoplist(stoplist=[str(i) for i in xrange(0,50,7)])

    e = BeagleEnvironment(ec, n_cols=5)
    e.train()

    cm = BeagleContextSeq(cc, ec, e.matrix)
    cm.train()

    om = BeagleOrderSeq(ec, e.matrix)
    om.train()

    m = BeagleComposite(cc, cm.matrix, ec, om.matrix)
    m.train()

    from tempfile import NamedTemporaryFile
    import os

    try:
        tmp = NamedTemporaryFile(delete=False, suffix='.npz')
        m.save(tmp.name)
        tmp.close()
        m1 = BeagleComposite.load(tmp.name)
        assert (m.matrix == m1.matrix).all()
    
    finally:
        os.remove(tmp.name)

    return m.matrix
