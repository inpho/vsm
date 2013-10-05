import unittest2 as unittest
import numpy as np

from vsm.model.beaglecontext import *


class TestBeagleContext(unittest.TestCase):

    def setUp(self):

        from vsm.corpus.util.corpusbuilders import random_corpus
        from vsm.model.beagleenvironment import BeagleEnvironment

        self.ec = random_corpus(100000, 5000, 0, 20, context_type='sentence')
        self.cc = self.ec.apply_stoplist(
                stoplist=[str(i) for i in xrange(0,5000,100)])

        self.e = BeagleEnvironment(self.ec, n_cols=200)
        self.e.train()

        self.ms = BeagleContextSeq(self.cc, self.ec, self.e.matrix)
        self.ms.train()

        self.mm = BeagleContextMulti(self.cc, self.ec, self.e.matrix)
        self.mm.train(n_procs=3)


    def test_BeagleContextSeq(self):
        
        from tempfile import NamedTemporaryFile
        import os

        try:
            tmp = NamedTemporaryFile(delete=False, suffix='.npz')
            self.ms.save(tmp.name)
            tmp.close()
            m1 = BeagleContextSeq.load(tmp.name)
            self.assertTrue((self.ms.matrix == m1.matrix).all())
    
        finally:
            os.remove(tmp.name)

        return self.ms.matrix


    def test_BeagleContextMulti(self):

        from tempfile import NamedTemporaryFile
        import os

        try:
            tmp = NamedTemporaryFile(delete=False, suffix='.npz')
            self.mm.save(tmp.name)
            tmp.close()
            m1 = BeagleContextMulti.load(tmp.name)
            self.assertTrue((self.mm.matrix == m1.matrix).all())
    
        finally:
            os.remove(tmp.name)

        return self.mm.matrix


    def test_compare(self):

        print 'Training single processor model'
        ms = BeagleContextSeq(self.cc, self.ec, self.e.matrix)
        ms.train()

        print 'Training multiprocessor model'
        mm = BeagleContextMulti(self.cc, self.ec, self.e.matrix)
        mm.train()

        self.assertTrue(np.allclose(ms.matrix, mm.matrix))
           
        
#Define and run test suite
suite = unittest.TestLoader().loadTestsFromTestCase(TestBeagleContext)
unittest.TextTestRunner(verbosity=2).run(suite)
