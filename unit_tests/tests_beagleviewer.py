import unittest2 as unittest
import numpy as np

from vsm.viewer.beagleviewer import BeagleViewer 
from vsm.viewer.labeleddata import *


class TestBeagleViewer(unittest.TestCase):

    def setUp(self):

        from vsm.corpus.util.corpusbuilders import random_corpus
        from vsm.model.beagleenvironment import BeagleEnvironment
        from vsm.model.beaglecontext import BeagleContextSeq
        from vsm.model.beagleorder import BeagleOrderSeq
        from vsm.model.beaglecomposite import BeagleComposite

        ec = random_corpus(1000, 50, 0, 20, context_type='sentence')
        cc = ec.apply_stoplist(stoplist=[str(i) for i in xrange(0,50,7)])
        e = BeagleEnvironment(ec, n_cols=5)
        e.train()

        cm = BeagleContextSeq(cc, ec, e.matrix)
        cm.train()

        om = BeagleOrderSeq(ec, e.matrix)
        om.train()

        m = BeagleComposite(cc, cm.matrix, ec, om.matrix)
        m.train()

        self.venv = BeagleViewer(ec, e)
        self.vctx = BeagleViewer(cc, cm)
        self.vord = BeagleViewer(ec, om)
        self.vcom = BeagleViewer(cc, m)


    def test_BeagleViewer(self):
    
        sww = self.venv.dist_word_word('1')        
        sww1 = self.vord.dist_word_word('0')
        self.assertTrue(type(sww) == LabeledColumn)
        self.assertTrue(type(sww1) == LabeledColumn)
        
        smw = self.vcom.dismat_word(['1'])
        smw1 = self.vctx.dismat_word(['1'])
        self.assertTrue(type(smw) == IndexedSymmArray)
        self.assertTrue(type(smw1) == IndexedSymmArray)


        
#Define and run test suite
suite = unittest.TestLoader().loadTestsFromTestCase(TestBeagleViewer)
unittest.TextTestRunner(verbosity=2).run(suite)
