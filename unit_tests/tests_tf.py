import unittest2 as unittest
import numpy as np

from vsm.model import tf
from multiprocessing import Process
import platform

class MPTester:
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

    def test_TfMulti_train(self):
        self.setUp()
        m = tf.TfMulti()
        m.corpus = self.corpus
        m.docs = self.docs
        m.V = self.V
        m.train(2)

        assert (self.cnt_mat == m.matrix.toarray()).all()

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
        
    def test_TF_proper_class(self):
        m = tf.TF(multiprocessing=True)
        if platform.system() == 'Windows':
            self.assertTrue(isinstance(m,tf.TfSeq))
        else:
            self.assertTrue(isinstance(m,tf.TfMulti))

    def test_TfSeq_train(self):
        m = tf.TfSeq()
        m.corpus = self.corpus
        m.docs = self.docs
        m.V = self.V
        m.train()
        self.assertTrue((self.cnt_mat == m.matrix.toarray()).all())

    def test_demo_TfMulti_train(self):
        t = MPTester()
        p = Process(target=t.test_TfMulti_train, args=())
        p.start()
        p.join()


        
#Define and run test suite
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTf)
    unittest.TextTestRunner(verbosity=2).run(suite)
