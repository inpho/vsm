import unittest2 as unittest
import numpy as np

from vsm.viewer.wrappers import *
from vsm.viewer.labeleddata import *




class TestViewerWrappers(unittest.TestCase):

    # TODO: Rewrite these to be independent of LDA
    pass

    # def setUp(self):

    #     from vsm.corpus.util.corpusbuilders import random_corpus
    #     from vsm.model.ldacgsseq import LdaCgsSeq

    #     self.c = random_corpus(1000, 50, 0, 20, context_type='sentence',
    #                         metadata=True)

    #     self.m = LDAGibbs(self.c, 'sentence', K=40)
    #     self.m.train(n_iterations=50)


    # def test_dist_(self):
    
    #     li = [0,1]

    #     swt = dist_word_top(self.c, self.m.word_top.T, '0')
    #     swtl = dist_word_top(self.c, self.m.word_top.T, ['0','1'], order='i')
    #     sww = dist_word_word(self.c, self.m.word_top, '0')
    #     swwl = dist_word_word(self.c, self.m.word_top, ['0','1'], order='i')
    #     std = dist_top_doc(self.c, self.m.top_doc.T, 0, 'sentence', order='i')
    #     stdl = dist_top_doc(self.c, self.m.top_doc.T, li, 'sentence')
    #     sdd = dist_doc_doc(self.c, self.m.top_doc, self.m.context_type, 0)
    #     sddl = dist_doc_doc(self.c, self.m.top_doc, self.m.context_type, li)
    #     stt = dist_top_top(self.m.word_top.T, 1)
    #     sttl = dist_top_top(self.m.word_top.T, li)

    #     self.assertEqual(type(swt), LabeledColumn)
    #     self.assertEqual(type(swtl), LabeledColumn)
    #     self.assertEqual(type(sww), LabeledColumn)
    #     self.assertEqual(type(swwl), LabeledColumn)
    #     self.assertEqual(type(std), LabeledColumn)
    #     self.assertEqual(type(stdl), LabeledColumn)
    #     self.assertEqual(type(sdd), LabeledColumn)
    #     self.assertEqual(type(sddl), LabeledColumn)
    #     self.assertEqual(type(stt), LabeledColumn)
    #     self.assertEqual(type(sttl), LabeledColumn)
        
        
    # def test_dismat_(self):

    #     dismatw = dismat_word(['0','2','5'], self.c, self.m.word_top)
    #     dismatd = dismat_doc([0,1,2], self.c, self.m.context_type, 
    #                          self.m.top_doc)
    #     dismatt = dismat_top([0,1,2], self.m.word_top)

    #     self.assertEqual(type(dismatw), IndexedSymmArray)
    #     self.assertEqual(type(dismatd), IndexedSymmArray)
    #     self.assertEqual(type(dismatt), IndexedSymmArray)
        
        

#Define and run test suite
suite = unittest.TestLoader().loadTestsFromTestCase(TestViewerWrappers)
unittest.TextTestRunner(verbosity=2).run(suite)
