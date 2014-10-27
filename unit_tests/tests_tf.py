import unittest2 as unittest
import numpy as np

from vsm.model import tf


class TestTf(unittest.TestCase):

    def setUp(self):
        self.corpus = np.array([0, 1, 3, 1, 1, 0, 3, 0, 3,
                                3, 0, 1, 0,
                                1, 3])
        self.docs = [slice(0,9), slice(9,13),
                     slice(13,13), slice(13,15)]
        self.V = 4
        self.cnt_mat = np.array([[3, 2, 0, 0],
                                 [3, 1, 0, 1],
                                 [0, 0, 0, 0],
                                 [3, 1, 0, 1]])
        

    def test_TfSeq_train(self):
        m = tf.TfSeq()
        m.corpus = self.corpus
        m.docs = self.docs
        m.V = self.V
        m.train()
        self.assertTrue((self.cnt_mat == m.matrix.toarray()).all())


    def test_TfMulti_train(self):
        m = tf.TfMulti()
        m.corpus = self.corpus
        m.docs = self.docs
        m.V = self.V
        m.train(2)

        self.assertTrue((self.cnt_mat == m.matrix.toarray()).all())


        
#Define and run test suite
suite = unittest.TestLoader().loadTestsFromTestCase(TestTf)
unittest.TextTestRunner(verbosity=2).run(suite)
