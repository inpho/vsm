import unittest2 as unittest
import numpy as np
from scipy.sparse import coo_matrix
from vsm.model import tfidf


class TestTfIdf(unittest.TestCase):

    def setUp(self):
        self.tf_mat = coo_matrix(np.array([[3, 2, 0, 0],
                                           [3, 1, 0, 1],
                                           [0, 0, 0, 0],
                                           [3, 1, 1, 1]], dtype=np.int))
        self.tfidf_mat = np.array(\
            [[2.0794415, 1.3862944, 0, 0],
             [0.86304623, 0.28768209, 0, 0.28768209],
             [0, 0, 0, 0],
             [0, 0, 0, 0]])
        self.undefined_rows = [2]
        

    def test_TfIdf_train(self):
        m = tfidf.TfIdf()
        m.train()
        self.assertTrue(m.matrix.size == 0)
        self.assertTrue(len(m.undefined_rows) == 0)
        
        m = tfidf.TfIdf(self.tf_mat)
        m.train()
        np.testing.assert_almost_equal(self.tfidf_mat, m.matrix.toarray())
        self.assertEqual(m.undefined_rows, self.undefined_rows)


        
#Define and run test suite
suite = unittest.TestLoader().loadTestsFromTestCase(TestTfIdf)
unittest.TextTestRunner(verbosity=2).run(suite)
