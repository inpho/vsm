import unittest2 as unittest
import numpy as np

from vsm.model.beagleenvironment import *



class TestBeagleEnvironment(unittest.TestCase):

    def setUp(self):

        from vsm.extensions.corpusbuilders import random_corpus

        c = random_corpus(1000, 100, 0, 20)

        self.m = BeagleEnvironment(c, n_cols=100)
        self.m.train()


    def test_BeagleEnvironment(self):
    
        self.assertTrue((self.m.matrix <= 1).all())
        self.assertTrue((self.m.matrix >= -1).all())

        norms = (self.m.matrix**2).sum(1)**0.5

        self.assertTrue(np.allclose(np.ones(norms.shape[0]), norms))


    def test_BE_IO(self):
        from tempfile import NamedTemporaryFile
        import os

        try:
            tmp = NamedTemporaryFile(delete=False, suffix='.npz')
            self.m.save(tmp.name)
            tmp.close()
            m1 = BeagleEnvironment.load(tmp.name)
            self.assertTrue((self.m.matrix == m1.matrix).all())
    
        finally:
            os.remove(tmp.name)
       
        
#Define and run test suite
suite = unittest.TestLoader().loadTestsFromTestCase(TestBeagleEnvironment)
unittest.TextTestRunner(verbosity=2).run(suite)
