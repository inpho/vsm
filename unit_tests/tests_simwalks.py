import unittest2 as unittest
import numpy as np

from vsm.viewer.simwalks import *
from vsm.viewer.labeleddata import *


class TestSimwalks(unittest.TestCase):

    def setUp(self):

        self.go_list = ['a', 'b', 'c', 'd']
       

    def test_bf_sim_walk(self):

        results = bf_sim_walk(test_sim_fn, 'd', self.go_list, 4)
        self.assertEqual(set(self.go_list), set(results))


    def test_df_sim_walk(self):

        results = df_sim_walk(test_sim_fn, 'd', self.go_list, 5, 4)
        self.assertEqual(set(self.go_list), set(results))
 
   

#Define and run test suite
suite = unittest.TestLoader().loadTestsFromTestCase(TestSimwalks)
unittest.TextTestRunner(verbosity=2).run(suite)
