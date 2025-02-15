from __future__ import division
from past.utils import old_div
import unittest
import numpy as np

from vsm.spatial import *

#TODO: add tests for recently added methods.
def KL(p,q):
    return sum(p*np.log2(old_div(p,q)))
def partial_KL(p,q):
    return p * np.log2(old_div((2*p), (p+q)))
def JS(p,q):
    return 0.5*(KL(p,((p+q)*0.5)) + KL(q,((p+q)*0.5)))
def JSD(p,q):
    return (0.5*(KL(p,((p+q)*0.5)) + KL(q,((p+q)*0.5))))**0.5


class TestSpatial(unittest.TestCase):

    def setUp(self):
        # 2 random distributions
        self.p=np.random.random_sample((5,))
        self.q=np.random.random_sample((5,))

        # normalize
        self.p /= self.p.sum()
        self.q /= self.q.sum()

    def test_KL_div(self):
        self.assertTrue(np.allclose(KL_div(self.p,self.q), KL(self.p,self.q)))
        
    def test_JS_div(self):
        self.assertTrue(np.allclose(JS_div(self.p,self.q), JS(self.p,self.q)))
    
    def test_JS_dist(self):
        self.assertTrue(np.allclose(JS_dist(self.p,self.q), JSD(self.p,self.q)))


    def test_count_matrix(self):
    
        arr = [1, 2, 4, 2, 1]
        slices = [slice(0,1), slice(1, 3), slice(3,3), slice(3, 5)]
        m = 6
        result = coo_matrix([[0, 0, 0, 0],
                         [1, 0, 0, 1],
                         [0, 1, 0, 1],
                         [0, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 0]])
    
        self.assertTrue((result.toarray() == 
                        count_matrix(arr, slices, m).toarray()).all())

    


suite = unittest.TestLoader().loadTestsFromTestCase(TestSpatial)
unittest.TextTestRunner(verbosity=2).run(suite)
