import unittest2 as unittest
import numpy as np

from vsm import *
from vsm.split import *

class TestCore(unittest.TestCase):

    def test_mp_split_ls(self):

        l = [slice(0,0), slice(0,0), slice(0,0)]
        self.assertTrue(len(mp_split_ls(l, 1)) == 1)
        self.assertTrue((mp_split_ls(l, 1)[0] == l).all())
        self.assertTrue(len(mp_split_ls(l, 2)) == 2)
        self.assertTrue((mp_split_ls(l, 2)[0] == 
                        [slice(0,0), slice(0,0)]).all())
        self.assertTrue((mp_split_ls(l, 2)[1] == [slice(0,0)]).all())
        self.assertTrue(len(mp_split_ls(l, 3)) == 3)
        self.assertTrue((mp_split_ls(l, 3)[0] == [slice(0,0)]).all())
        self.assertTrue((mp_split_ls(l, 3)[1] == [slice(0,0)]).all())
        self.assertTrue((mp_split_ls(l, 3)[2] == [slice(0,0)]).all())

suite = unittest.TestLoader().loadTestsFromTestCase(TestCore)
unittest.TextTestRunner(verbosity=2).run(suite)
