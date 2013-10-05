import unittest2 as unittest
import numpy as np

from vsm.model.beaglecomposite import *


class TestBeagleComposite(unittest.TestCase):

    def setUp(self):
        from vsm.corpus.util.corpusbuilders import random_corpus
        from vsm.model.beagleenvironment import BeagleEnvironment
        from vsm.model.beaglecontext import BeagleContextSeq
        from vsm.model.beagleorder import BeagleOrderSeq

        ec = random_corpus(1000, 50, 0, 20, context_type='sentence')
        cc = ec.apply_stoplist(stoplist=[str(i) for i in xrange(0,50,7)])

        e = BeagleEnvironment(ec, n_cols=5)
        e.train()

        self.cm = BeagleContextSeq(cc, ec, e.matrix)
        self.cm.train()

        self.om = BeagleOrderSeq(ec, e.matrix)
        self.om.train()

        self.m = BeagleComposite(cc, self.cm.matrix, ec, self.om.matrix)
        self.m.train()


    def test_BeagleCompositeIO(self):
        from tempfile import NamedTemporaryFile
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
