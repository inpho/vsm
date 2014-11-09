import unittest2 as unittest
import numpy as np

from vsm.corpus.util.corpusbuilders import random_corpus
from vsm.model.ldacgsseq import *


class TestLdaCgsSeq(unittest.TestCase):

    def setUp(self):
        pass

    ##TODO: write actual test cases.

    def test_LdaCgsSeq_IO(self):

        from tempfile import NamedTemporaryFile
        import os
    
        c = random_corpus(1000, 50, 6, 100)
        tmp = NamedTemporaryFile(delete=False, suffix='.npz')
        try:
            m0 = LdaCgsSeq(c, 'document', K=10)
            m0.train(n_iterations=20)
            m0.save(tmp.name)
            m1 = LdaCgsSeq.load(tmp.name)
            self.assertTrue(m0.context_type == m1.context_type)
            self.assertTrue(m0.K == m1.K)
            self.assertTrue((m0.alpha == m1.alpha).all())
            self.assertTrue((m0.beta == m1.beta).all())
            self.assertTrue(m0.log_probs == m1.log_probs)
            for i in xrange(max(len(m0.corpus), len(m1.corpus))):
                self.assertTrue(m0.corpus[i].all() == m1.corpus[i].all())
            self.assertTrue(m0.V == m1.V)
            self.assertTrue(m0.iteration == m1.iteration)
            for i in xrange(max(len(m0.Z), len(m1.Z))):
                self.assertTrue(m0.Z[i].all() == m1.Z[i].all())
            self.assertTrue(m0.top_doc.all() == m1.top_doc.all())
            self.assertTrue(m0.word_top.all() == m1.word_top.all())
            self.assertTrue(m0.inv_top_sums.all() == m1.inv_top_sums.all())
            m0 = LdaCgsSeq(c, 'document', K=10)
            m0.train(n_iterations=20)
            m0.save(tmp.name)
            m1 = LdaCgsSeq.load(tmp.name)
            self.assertTrue(not hasattr(m1, 'log_prob'))
        finally:
            os.remove(tmp.name)


    def test_LdaCgsQuerySampler(self):

        
        pass



suite = unittest.TestLoader().loadTestsFromTestCase(TestLdaCgsSeq)
unittest.TextTestRunner(verbosity=2).run(suite)
