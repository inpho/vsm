import unittest2 as unittest
import numpy as np

from vsm.spatial import *

#TODO: add tests for recently added methods.
def KL(p,q):
    return sum(p*np.log2(p/q))
def partial_KL(p,q):
    return p * np.log2((2*p) / (p+q))
def JS(p,q):
    return 0.5*(KL(p,((p+q)*0.5)) + KL(q,((p+q)*0.5)))
def JSD(p,q):
    return (0.5*(KL(p,((p+q)*0.5)) + KL(q,((p+q)*0.5))))**0.5


class TestLinalg(unittest.TestCase):

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


    def test_KL_div_old(self):
        p = np.array([0,1])
        Q = np.array([[0,1],
                      [.5,.5],
                      [1,0]])
        
        exp = np.array([np.inf, 1, 0])

        self.assertTrue(np.allclose(exp, KL_div(p,Q)))

    '''
    def test_row_cos_mat(self):

        m = np.random.random((10,5))
        out_1 = np.zeros((10,10))
        for i, j in zip(*np.triu_indices_from(out_1)):
            out_1[i, j] = (np.dot(m[i], m[j])
                            / (np.dot(m[i], m[i])**.5
                            * np.dot(m[j], m[j])**.5))
        out_2 = row_cos_mat(range(10), m, fill_tril=False)
        
        self.assertTrue(np.allclose((out_1, out_2), (out_1, out_2)))

    def test_hstack_coo(self):

        dense_mat_ls = [np.random.random((3,4)),
                        np.random.random((3,5)),
                        np.random.random((3,6))]

        mat_ls = [coo_matrix(m) for m in dense_mat_ls]
    
        self.assertTrue((np.hstack(dense_mat_ls) == 
                        hstack_coo(mat_ls).toarray()).all())
    '''

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

    


suite = unittest.TestLoader().loadTestsFromTestCase(TestLinalg)
unittest.TextTestRunner(verbosity=2).run(suite)
