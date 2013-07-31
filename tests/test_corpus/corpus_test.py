import unittest
import numpy as np
import os
from vsm.corpus import *
from tempfile import NamedTemporaryFile

class TestCorpus(unittest.TestCase):

    def setUp(self):
        corpus = np.array([0, 3, 2, 1, 0, 3, 0, 2, 3, 0, 2, 3, 1, 2, 0, 3,
                                2, 1, 2, 2], dtype=np.int)
        contextData = np.array([(3, 'doc0'), (5, 'doc1'), (7,'doc2'), (11,'doc3'),
                (11,'doc4'), (15,'doc5'), (18,'doc6'), (20,'doc7')], 
                dtype=[('idx', '<i8'), ('doc', '|S4')])

        self.bc = BaseCorpus(corpus, context_data=[contextData],
                                     context_types=['document'])
        
        text = ['I', 'came', 'I', 'saw', 'I', 'conquered']
        ctx_data = [np.array([(2, 'Veni'), (4, 'Vidi'), (6, 'Vici')],
                            dtype=[('idx', '<i8'), ('sent', '|S6')])]

        self.corpus = Corpus(text, context_data=ctx_data,
                                    context_types=['sentence'])
        
                            
    def test_SplitCorpus(self): 
        odd = split_corpus(self.bc.corpus, [1,3,5])
        even = split_corpus(self.bc.corpus, [0,2,4])

        odd_split = (np.array([1,0,2,3,0,2,3,1,2,0]) == odd).all()
        even_split = (np.array([0,3,2,3,0]) == even).all()

        self.assertTrue(odd, msg=None)
        self.assertEqual(even, msg=None)

    def test_ValidateIndices(self):
        for t in contextData:
            self.assertTrue(self.bc._validate_indices(t['idx']))

    def test_RemoveEmpty(self):
        self.bc.remove_empty()
        new_ctx = (self.bc.context_data == np.array([(3,'doc0'), (5,'doc1'), 
                (7,'doc2'), (11,'doc3'), (15,'doc5'), (18,'doc6'), 
                (20,'doc7')])).all()

        self.assertTrue(new_ctx, msg=None)

    def test_ViewMetadata(self):
        meta = self.bc.view_metadata('document')
        self.assertEqual(self.bc.context_data[0], meta)

    def test_bc_ViewContexts(self):
        ctx = self.bc.view_contexts('document')
        self.assertEqual([array([0,3,2]), array([1,0]), array([3,0]),
            array([2,3,0,2]), array([]), array([3,1,2,0]), array([3,2,1]),
            array([2,2])], ctx)

    
    def test_MetaInt(self):
        i = self.bc.meta_int('document', {'doc': 'doc3'})
        self.assertEqual(4, i)

    def test_GetMetadum(self):
        s = self.bc.get_metadum('document', {'doc': 'doc0'}, 'doc')
        self.assertEqual('doc0', s)

   
    def test_SetWordsInt(self):
        d = {0:'I', 1:'came', 2:'saw', 3:'conquered'}
        self.assertEqual(self.corpus.words_int, d)

    def test_ViewContexts(self):
        ctx = [array(['I','came']), array(['I', 'saw']), array(['I', 'conquered'])]
        self.assertEqual(self.corpus.view_contexts('sentence'), ctx)

    def test_SaveLoad(self):
        
        try:
            tmp = NamedTemporaryFile(delete=False, suffix='.npz')
            self.corpus.save(tmp.name)
            tmp.close()
            c_reloaded = self.corpus.load(tmp.name)

            assert (self.corpus.corpus == c_reloaded.corpus).all()
            assert (self.corpus.words == c_reloaded.words).all()
            assert self.corpus.words_int == c_reloaded.words_int
            assert self.corpus.context_types == c_reloaded.context_types
            for i in xrange(len(self.corpus.context_data)):
                self.assertTrue((self.corpus.context_data[i] == 
                        c_reloaded.context_data[i]).all(), msg=None)
    
        finally:
            os.remove(tmp.name)


if __name__=='__main__':
    unittest.main()

