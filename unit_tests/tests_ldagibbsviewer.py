import unittest2 as unittest
import numpy as np

from vsm.viewer.ldagibbsviewer import *
from vsm.viewer.labeleddata import *


class TestLDAGibbsViewer(unittest.TestCase):

    def setUp(self):

        from vsm.corpus.util.corpusbuilders import random_corpus
        from vsm.model.ldagibbs import LDAGibbs

        c = random_corpus(1000, 50, 0, 20, context_type='sentence',
                            metadata=True)

        m = LDAGibbs(c, 'sentence', K=40)
        m.train(itr=50)

        self.ldav = LDAGibbsViewer(c, m)


    def test_LDAGibbsViewer(self):
    
        li = [0,1]

        t = self.ldav.topics(compact_view=False)
        te = self.ldav.topic_entropies(compact_view=False)
        swt = self.ldav.sim_word_top('0', compact_view=False)
        
        dt = self.ldav.doc_topics(0)
        dt_ = self.ldav.doc_topics(0)
        wt = self.ldav.word_topics('0')
        stt = self.ldav.sim_top_top(1)
        sttl = self.ldav.sim_top_top(li)
        std = self.ldav.sim_top_doc(0)
        stdl = self.ldav.sim_top_doc(li) 
        sww = self.ldav.sim_word_word('0')
        swwl = self.ldav.sim_word_word(['0','1'])
        sdd = self.ldav.sim_doc_doc(0)
        sddl = self.ldav.sim_doc_doc(li)
        
        t_c = self.ldav.topics()
        te_c = self.ldav.topic_entropies()
        swt_c = self.ldav.sim_word_top('1')
        
        simmatw = self.ldav.simmat_words(['0','2','5'])
        simmatd = self.ldav.simmat_docs()
        simmatt = self.ldav.simmat_topics()

        self.assertEqual(type(t), DataTable)
        self.assertEqual(type(te), DataTable)
        self.assertEqual(type(swt), DataTable)
        
        self.assertEqual(type(dt), LabeledColumn)
        self.assertEqual(type(dt_), LabeledColumn)
        self.assertEqual(type(wt), LabeledColumn)
        self.assertEqual(type(stt), LabeledColumn)
        self.assertEqual(type(sttl), LabeledColumn)
        self.assertEqual(type(std), LabeledColumn)
        self.assertEqual(type(stdl), LabeledColumn)
        self.assertEqual(type(sww), LabeledColumn)
        self.assertEqual(type(swwl), LabeledColumn)
        self.assertEqual(type(sdd), LabeledColumn)
        self.assertEqual(type(sddl), LabeledColumn)
        
        self.assertEqual(type(t_c), CompactTable)
        self.assertEqual(type(te_c), CompactTable)
        self.assertEqual(type(swt_c), CompactTable)

        self.assertEqual(type(simmatw), IndexedSymmArray)
        self.assertEqual(type(simmatd), IndexedSymmArray)
        self.assertEqual(type(simmatt), IndexedSymmArray)

        

#Define and run test suite
suite = unittest.TestLoader().loadTestsFromTestCase(TestLDAGibbsViewer)
unittest.TextTestRunner(verbosity=2).run(suite)
