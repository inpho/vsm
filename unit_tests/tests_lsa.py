import unittest2 as unittest
import numpy as np

from vsm.model.lsa import Lsa


class TestLsa(unittest.TestCase):

    def setUp(self):

        self.tfidf_mat = np.array(\
            [[2.0794415, 1.3862944, 0],
             [0.86304623, 0.28768209, 0.28768209],
             [np.inf, np.inf, np.inf],
             [0, 0, 0]])
        self.eigenvalues = np.array(\
            [ 0.35270742,  2.65176495])
        self.doc_matrix = np.array([0.314334, 0.023485])
    
    #TODO: Write some actual unit tests for this module

        
#Define and run test suite
suite = unittest.TestLoader().loadTestsFromTestCase(TestLsa)
unittest.TextTestRunner(verbosity=2).run(suite)
