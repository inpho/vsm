import numpy as np

from vsm import row_normalize
from vsm.model.beaglecontext import realign_env_mat



class BeagleComposite(object):

    def __init__(self, ctx_corp, ctx_matrix, ord_corp, ord_matrix):
        """
        Assume that the context corpus is a subcorpus of the order
        corpus and that the eventual composite corpus is the context
        corpus. The order matrix is sliced and reordered so that it
        aligns with the context matrix.
        """
        self.ctx_matrix = row_normalize(ctx_matrix)
        ord_matrix = realign_env_mat(ctx_corp, ord_corp, ord_matrix)
        self.ord_matrix = row_normalize(ord_matrix)


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

    return m.matrix
