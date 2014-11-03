import unittest2 as unittest
import numpy as np

from vsm.viewer.similarity import *
from vsm.viewer.labeleddata import *


class TestSimilarity(unittest.TestCase):

    def setUp(self):

        from vsm.corpus.util.corpusbuilders import random_corpus
        from vsm.model.ldagibbs import LDAGibbs

        self.c = random_corpus(1000, 50, 0, 20, context_type='sentence',
                            metadata=True)

        self.m = LDAGibbs(self.c, 'sentence', K=40)
        self.m.train(itr=50)


    def test_sim_(self):
    
        li = [0,1]

        swt = sim_word_top(self.c, self.m.top_word, '0')
        swtl = sim_word_top(self.c, self.m.top_word, ['0','1'], order='i')
        sww = sim_word_word(self.c, self.m.top_word.T, '0')
        swwl = sim_word_word(self.c, self.m.top_word.T, ['0','1'], order='i')
        std = sim_top_doc(self.c, self.m.doc_top, 0, 'sentence', order='i')
        stdl = sim_top_doc(self.c, self.m.doc_top, li, 'sentence')
        sdd = sim_doc_doc(self.c, self.m.doc_top.T, self.m.context_type, 0)
        sddl = sim_doc_doc(self.c, self.m.doc_top.T, self.m.context_type, li)
        stt = sim_top_top(self.m.top_word, 1)
        sttl = sim_top_top(self.m.top_word, li)

        self.assertEqual(type(swt), LabeledColumn)
        self.assertEqual(type(swtl), LabeledColumn)
        self.assertEqual(type(sww), LabeledColumn)
        self.assertEqual(type(swwl), LabeledColumn)
        self.assertEqual(type(std), LabeledColumn)
        self.assertEqual(type(stdl), LabeledColumn)
        self.assertEqual(type(sdd), LabeledColumn)
        self.assertEqual(type(sddl), LabeledColumn)
        self.assertEqual(type(stt), LabeledColumn)
        self.assertEqual(type(sttl), LabeledColumn)
        
        
    def test_simmat_(self):

        simmatw = simmat_words(self.c, self.m.top_word.T, ['0','2','5'])
        simmatd = simmat_documents(self.c, self.m.doc_top.T, 
                            self.m.context_type, [0,1,2])
        simmatt = simmat_topics(self.m.top_word, [0,1,2])

        self.assertEqual(type(simmatw), IndexedSymmArray)
        self.assertEqual(type(simmatd), IndexedSymmArray)
        self.assertEqual(type(simmatt), IndexedSymmArray)
        
        

#Define and run test suite
suite = unittest.TestLoader().loadTestsFromTestCase(TestSimilarity)
unittest.TextTestRunner(verbosity=2).run(suite)
