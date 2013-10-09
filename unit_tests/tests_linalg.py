import unittest2 as unittest
import numpy as np

from vsm.linalg import *

#TODO: add tests for recently added methods.

class TestLinalg(unittest.TestCase):

    def test_row_normalize(self):

        m = np.random.random((5,7))
        m = row_normalize(m)

        self.assertTrue(np.allclose(row_norms(m), np.ones(5)))


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
