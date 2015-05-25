import unittest2 as unittest
import numpy as np
import os
from vsm.corpus import *
from vsm.split import split_corpus
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
        
                        
    #TODO: Move this test to vsm.split
    def test_SplitCorpus(self): 
        odd = split_corpus(self.corpus.corpus, [1,3,5])
        even = split_corpus(self.corpus.corpus, [2,4,6])

        odd_expected = [np.array([0]), np.array([1, 0]),
                np.array([3, 0]), np.array([2])]
        even_expected = [np.array([0, 1]), np.array([0, 3]),
                np.array([0, 2])]

        for i in xrange(len(odd)):
            np.testing.assert_array_equal(odd[i], odd_expected[i])
        for i in xrange(len(even)):
            np.testing.assert_array_equal(even[i], even_expected[i])


    def test_align_corpora(self):
        
        out = align_corpora(self.corpus, Corpus([]))
        self.assertTrue(len(out.corpus)==0)
        self.assertTrue(len(out.words)==4)
        self.assertTrue(len(out.words_int)==4)

        out = align_corpora(Corpus([], remove_empty=False), self.corpus)
        self.assertTrue(len(out.corpus)==0)
        self.assertTrue(len(out.words)==0)
        self.assertTrue(len(out.words_int)==0)

        out = align_corpora(self.corpus, self.corpus)
        self.assertTrue(len(out.corpus)==len(self.corpus.corpus))
        self.assertTrue((out.corpus==self.corpus.corpus).all())
        self.assertTrue(len(out.words)==len(self.corpus.words))
        self.assertTrue((out.words==self.corpus.words).all())
        self.assertTrue(out.words_int==self.corpus.words_int)

        new_corp = Corpus(
            [ 'came', 'saw', 'and', 'conquered' ],
            context_data=[ np.array([(4, )], dtype=[('idx', '<i8')]) ])
        out = align_corpora(self.corpus, new_corp)
        self.assertTrue(len(out.corpus)==3)
        for w in out.corpus:
            self.assertTrue(out.words[w]==self.corpus.words[w])
        self.assertTrue(len(out.words)==4)
        self.assertTrue((out.words==self.corpus.words).all())
        self.assertTrue(out.words_int==self.corpus.words_int)


    def test_ValidateIndices(self):
        for t in self.bc.context_data:
            self.assertTrue(self.bc._validate_indices(t['idx']))

    def test_RemoveEmpty(self):
        self.bc.remove_empty()
        new_ctx = np.equal(self.bc.context_data[0], np.array([(3,'doc0'), (5,'doc1'), 
                (7,'doc2'), (11,'doc3'), (15,'doc5'), (18,'doc6'), 
                (20,'doc7')], dtype=[('idx', '<i8'), ('sent', '|S6')]))
        self.assertTrue(new_ctx, msg=None)

    def test_ViewMetadata(self):
        meta = self.bc.view_metadata('document')
        np.testing.assert_array_equal(self.bc.context_data[0], meta)

    def test_bc_ViewContexts(self):
        ctx = self.bc.view_contexts('document')
        expected = [np.array([0,3,2]), np.array([1,0]), np.array([3,0]),
             np.array([2,3,0,2]), np.array([3,1,2,0]), np.array([3,2,1]),
             np.array([2,2])]
        for i in xrange(len(ctx)):
            np.testing.assert_array_equal(ctx[i], expected[i])
        
    
    def test_MetaInt(self):
        i = self.bc.meta_int('document', {'doc': 'doc3'})
        self.assertEqual(3, i)

    def test_GetMetadatum(self):
        s = self.bc.get_metadatum('document', {'doc': 'doc0'}, 'doc')
        self.assertEqual('doc0', s)

   
    def test_SetWordsInt(self):
        d = {'I':0, 'came':1, 'conquered':2, 'saw':3}
        self.assertEqual(self.corpus.words_int, d)

    def test_ViewContexts(self):
        expected = [np.array(['I','came']), np.array(['I', 'saw']), np.array(['I', 'conquered'])]
        ctx = self.corpus.view_contexts('sentence', as_strings=True)
        for i in xrange(len(ctx)):
            np.testing.assert_array_equal(ctx[i], expected[i])

    def test_SaveLoad(self):
        
        try:
            tmp = NamedTemporaryFile(delete=False, suffix='.npz')
            self.corpus.save(tmp.name)
            tmp.close()
            c_reloaded = Corpus.load(tmp.name)

            self.assertTrue((self.corpus.corpus == c_reloaded.corpus).all())
            self.assertTrue((self.corpus.words == c_reloaded.words).all())
            self.assertTrue(self.corpus.words_int == c_reloaded.words_int)
            self.assertTrue(self.corpus.context_types == c_reloaded.context_types)
            
            for i in xrange(len(self.corpus.context_data)):
                self.assertTrue((self.corpus.context_data[i] == 
                        c_reloaded.context_data[i]).all(), msg=None)
    
        finally:
            os.remove(tmp.name)


suite = unittest.TestLoader().loadTestsFromTestCase(TestCorpus)
unittest.TextTestRunner(verbosity=2).run(suite)
