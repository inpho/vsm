import unittest2 as unittest
import numpy as np

from vsm.corpus.util.corpusbuilders import random_corpus
from vsm.model.ldagibbs import LDAGibbs


class TestLdagibbs(unittest.TestCase):

    def setUp(self):
        self.c = random_corpus(500000, 10000, 0, 100)
        self.m = LDAGibbs(self.c, 'context', K=20)


    ##TODO: polish up test cases.
    def test_z_dist(self):
        np.random.seed(0)
        self.m.train(itr=1)

        return self.m

    
    def test_logp_fns(self):
        self.m.train(itr=1)
    
        logp_1 = self.m._logp()
        logp_2 = self.m.logp()

        self.assertTrue(np.allclose(logp_1, logp_2), (logp_1,logp_2))

        return self.m


    def test_LDAGibbs_IO(self):

        from tempfile import NamedTemporaryFile
        import os
    
        c = random_corpus(1000, 50, 6, 100)
        tmp = NamedTemporaryFile(delete=False, suffix='.npz')
        try:
            m0 = LDAGibbs(c, 'context', K=10)
            m0.train(itr=20)
            m0.save(tmp.name)
            m1 = LDAGibbs.load(tmp.name)
            self.assertTrue(m0.context_type == m1.context_type)
            self.assertTrue(m0.K == m1.K)
            self.assertTrue(m0.alpha == m1.alpha)
            self.assertTrue(m0.beta == m1.beta)
            self.assertTrue(m0.log_prob == m1.log_prob)
            for i in xrange(max(len(m0.W), len(m1.W))):
                self.assertTrue(m0.W[i].all() == m1.W[i].all())
            self.assertTrue(m0.V == m1.V)
            self.assertTrue(m0.iterations == m1.iterations)
            for i in xrange(max(len(m0.Z), len(m1.Z))):
                self.assertTrue(m0.Z[i].all() == m1.Z[i].all())
            self.assertTrue(m0.doc_top.all() == m1.doc_top.all())
            self.assertTrue(m0.top_word.all() == m1.top_word.all())
            self.assertTrue(m0.sum_word_top.all() == m1.sum_word_top.all())
            m0 = LDAGibbs(c, 'context', K=10, log_prob=False)
            m0.train(itr=20)
            m0.save(tmp.name)
            m1 = LDAGibbs.load(tmp.name)
            self.assertTrue(not hasattr(m1, 'log_prob'))
        finally:
            os.remove(tmp.name)


suite = unittest.TestLoader().loadTestsFromTestCase(TestLdagibbs)
unittest.TextTestRunner(verbosity=2).run(suite)
