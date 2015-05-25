import unittest2 as unittest
import numpy as np

from vsm.viewer.ldacgsviewer import *
from vsm.viewer.labeleddata import *


class TestLdaCgsViewer(unittest.TestCase):

    def setUp(self):

        from vsm.corpus.util.corpusbuilders import random_corpus
        from vsm.model.ldacgsseq import LdaCgsSeq

        c = random_corpus(1000, 50, 0, 20, context_type='document',
                            metadata=True)

        m = LdaCgsSeq(c, 'document', K=10)
        m.train(n_iterations=50, verbose=0)

        self.ldav = LdaCgsViewer(c, m)


    def test_LdaCgsViewer(self):
    
        li = [0,1]

        t = self.ldav.topics(compact_view=False)
        te = self.ldav.topic_entropies()
        swt = self.ldav.dist_word_top('0', compact_view=False)
        
        dt = self.ldav.doc_topics(0)
        dt_ = self.ldav.doc_topics(0)
        wt = self.ldav.word_topics('0')
        stt = self.ldav.dist_top_top(1)
        sttl = self.ldav.dist_top_top(li)
        std = self.ldav.dist_top_doc(0)
        stdl = self.ldav.dist_top_doc(li) 
        sdd = self.ldav.dist_doc_doc(0)
        sddl = self.ldav.dist_doc_doc(li)
        
        t_c = self.ldav.topics()
        te_c = self.ldav.topic_entropies()
        swt_c = self.ldav.dist_word_top('1')
        
        dismatd = self.ldav.dismat_doc()
        dismatt = self.ldav.dismat_top()

        self.assertEqual(type(t), DataTable)
        self.assertEqual(type(te), LabeledColumn)
        self.assertEqual(type(swt), DataTable)
        
        self.assertEqual(type(dt), LabeledColumn)
        self.assertEqual(type(dt_), LabeledColumn)
        self.assertEqual(type(wt), LabeledColumn)
        self.assertEqual(type(stt), DataTable)
        self.assertEqual(type(sttl), DataTable)
        self.assertEqual(type(std), LabeledColumn)
        self.assertEqual(type(stdl), LabeledColumn)
        self.assertEqual(type(sdd), LabeledColumn)
        self.assertEqual(type(sddl), LabeledColumn)
        
        self.assertEqual(type(t_c), DataTable)
        self.assertEqual(type(te_c), LabeledColumn)
        self.assertEqual(type(swt_c), DataTable)

        self.assertEqual(type(dismatd), IndexedSymmArray)
        self.assertEqual(type(dismatt), IndexedSymmArray)

    def test_LdaCgsViewer_topics_args(self):
        # test calls of ldav.topics()
        t = self.ldav.topics()
        self.assertEqual(type(t), DataTable)
        self.assertEqual(len(t), self.ldav.model.K)

        with self.assertRaises(ValueError):
            self.ldav.topics(2)
        
        t=self.ldav.topics([2])
        self.assertEqual(type(t), DataTable)
        self.assertEqual(len(t), 1)

        t = self.ldav.topics([2,4])
        self.assertEqual(type(t), DataTable)
        self.assertEqual(len(t), 2)

        

#Define and run test suite
suite = unittest.TestLoader().loadTestsFromTestCase(TestLdaCgsViewer)
unittest.TextTestRunner(verbosity=2).run(suite)
