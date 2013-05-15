import unittest2 as unittest
import numpy as np

from vsm.model import lsa


class TestLsa(unittest.TestCase):

    def setUp(self):
        self.tfidf_mat = np.array(\
            [[2.0794415, 1.3862944, 0],
             [0.86304623, 0.28768209, 0.28768209],
             [np.inf, np.inf, np.inf],
             [0, 0, 0]])
        self.eigenvalues = np.array(\
            [ 0.35270742,  2.65176495])


    def test_Lsa_train(self):
        m = lsa.Lsa()
        m.train()
        self.assertTrue(m.word_matrix.size == 0)
        self.assertTrue(m.doc_matrix.size == 0)
        self.assertTrue(m.eigenvalues.size == 0)
        
        m = lsa.Lsa(self.tfidf_mat)
        m.train()

        self.assertTrue(np.allclose(self.eigenvalues, m.eigenvalues))


        
#Define and run test suite
suite = unittest.TestLoader().loadTestsFromTestCase(TestLsa)
unittest.TextTestRunner(verbosity=2).run(suite)
