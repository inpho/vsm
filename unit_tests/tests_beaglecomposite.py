import unittest2 as unittest
import numpy as np


class TestBeagleComposite(unittest.TestCase):

    def setUp(self):
        from vsm.corpus.util.corpusbuilders import random_corpus
        from vsm.model.beaglecomposite import BeagleComposite 
        from vsm.model.beagleenvironment import BeagleEnvironment
        from vsm.model.beaglecontext import BeagleContextSeq
        from vsm.model.beagleorder import BeagleOrderSeq

        self.ec = random_corpus(1000, 50, 0, 20, context_type='sentence')
        self.cc = self.ec.apply_stoplist(stoplist=[str(i) for i in xrange(0,50,7)])

        self.e = BeagleEnvironment(self.ec, n_cols=5)
        self.e.train()

        self.cm = BeagleContextSeq(self.cc, self.ec, self.e.matrix)
        self.cm.train()

        self.om = BeagleOrderSeq(self.ec, self.e.matrix)
        self.om.train()

        self.m = BeagleComposite(self.cc, self.cm.matrix, self.ec, self.om.matrix)
        self.m.train()


    def test_BeagleCompositeIO(self):
        from tempfile import NamedTemporaryFile
        from vsm.model.beaglecomposite import BeagleComposite 
        import os

        try:
            tmp = NamedTemporaryFile(delete=False, suffix='.npz')
            self.m.save(tmp.name)
            tmp.close()
            m1 = BeagleComposite.load(tmp.name)
            self.assertTrue((self.m.matrix == m1.matrix).all())
    
        finally:
            os.remove(tmp.name)



        
#Define and run test suite
suite = unittest.TestLoader().loadTestsFromTestCase(TestBeagleComposite)
unittest.TextTestRunner(verbosity=2).run(suite)
