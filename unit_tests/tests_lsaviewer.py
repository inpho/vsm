import unittest2 as unittest
import numpy as np

from vsm.viewer.lsaviewer import LsaViewer 


class TestLsaViewer(unittest.TestCase):

    def setUp(self):
        
        from vsm.corpus.util.corpusbuilders import random_corpus
        from vsm.model.tf import TfSeq
        from vsm.model.tfidf import TfIdf
        from vsm.model.lsa import Lsa

        c = random_corpus(10000, 1000, 0, 30, context_type='document', metadata=True)

        tf = TfSeq(c, 'document')
        tf.train()

        tfidf = TfIdf.from_tf(tf)
        tfidf.train()

        m = Lsa.from_tfidf(tfidf)
        m.train()

        self.v = LsaViewer(c, m)


    def test_Lsaviewer(self):
        
        from vsm.viewer.labeleddata import LabeledColumn, IndexedSymmArray 

        sww = self.v.dist_word_word('1')
        swwl = self.v.dist_word_word(['1', '0'])
        sdd = self.v.dist_doc_doc(1)
        sddl = self.v.dist_doc_doc([1, 0])
        self.assertTrue(type(sww) == LabeledColumn)
        self.assertTrue(type(swwl) == LabeledColumn)
        self.assertTrue(type(sdd) == LabeledColumn)
        self.assertTrue(type(sddl) == LabeledColumn)

        sw = self.v.dismat_word(['2','4','5'])
        sd = self.v.dismat_doc([1, 0])
        self.assertTrue(type(sw) == IndexedSymmArray)
        self.assertTrue(type(sd) == IndexedSymmArray)



        
#Define and run test suite
suite = unittest.TestLoader().loadTestsFromTestCase(TestLsaViewer)
unittest.TextTestRunner(verbosity=2).run(suite)
