import unittest2 as unittest
import numpy as np

from vsm.viewer.tfidfviewer import *
from vsm.viewer.labeleddata import *


class TestTfIdfViewer(unittest.TestCase):

    def setUp(self):
    
        from vsm.corpus.util.corpusbuilders import random_corpus
        from vsm.model.tf import TfSeq
        from vsm.model.tfidf import TfIdf

        c = random_corpus(1000, 100, 0, 20, context_type='document', metadata=True)

        tf = TfSeq(c, 'document')
        tf.train()

        m = TfIdf(tf.matrix, 'document')
        m.train()

        self.v = TfIdfViewer(c, m)
    
    def test_TfIdfViewer(self):

        li = [0,1]

        sww = self.v.sim_word_word('0')
        swwl = self.v.sim_word_word(['0','1'])
        sdd = self.v.sim_doc_doc(0)
        sddl = self.v.sim_doc_doc(li)

        simmatw = self.v.simmat_words(['0','2','5'])
        simmatd = self.v.simmat_docs(li)

        self.assertEqual(type(sww), LabeledColumn)
        self.assertEqual(type(swwl), LabeledColumn)
        self.assertEqual(type(sdd), LabeledColumn)
        self.assertEqual(type(sddl), LabeledColumn)

        self.assertEqual(type(simmatw), IndexedSymmArray)
        self.assertEqual(type(simmatd), IndexedSymmArray)



#Define and run test suite
suite = unittest.TestLoader().loadTestsFromTestCase(TestTfIdfViewer)
unittest.TextTestRunner(verbosity=2).run(suite)
