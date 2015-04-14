import unittest2 as unittest
import numpy as np
from scipy.sparse import coo_matrix
from vsm.model import tfidf
from vsm.model import tf


class TestTfIdf(unittest.TestCase):

    def setUp(self):
        self.corpus = np.array([0, 1, 3, 1, 1, 0, 3, 0, 3,
                                3, 0, 1, 0,
                                3,
                                1, 3])
        self.docs = [slice(0,9), slice(9,13),
                     slice(13,14), slice(14,16)]
        self.V = 4

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
        
        m = tfidf.TfIdf(tf_matrix=self.tf_mat)
        m.train()
        np.testing.assert_almost_equal(self.tfidf_mat, m.matrix.toarray())
        self.assertEqual(m.undefined_rows, self.undefined_rows)

    def test_TfIdf_from_tf(self):
        tf_model = tf.TF()
        tf_model.corpus = self.corpus
        tf_model.docs = self.docs
        tf_model.V = self.V
        tf_model.train()
        self.assertTrue((self.tf_mat == tf_model.matrix.toarray()).all())

        m = tfidf.TfIdf.from_tf(tf_model)
        self.assertTrue((m.matrix == tf_model.matrix.toarray()).all())
        m.train()
        np.testing.assert_almost_equal(self.tfidf_mat, m.matrix.toarray())
        self.assertEqual(m.undefined_rows, self.undefined_rows)

        
#Define and run test suite
suite = unittest.TestLoader().loadTestsFromTestCase(TestTfIdf)
unittest.TextTestRunner(verbosity=2).run(suite)
