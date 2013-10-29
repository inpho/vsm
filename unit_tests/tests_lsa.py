import unittest2 as unittest
import numpy as np

from vsm.model import lsa


class TestLsa(unittest.TestCase):

    def setUp(self):

        from vsm.corpus.util.corpusbuilders import random_corpus

        self.tfidf_mat = np.array(\
            [[2.0794415, 1.3862944, 0],
             [0.86304623, 0.28768209, 0.28768209],
             [np.inf, np.inf, np.inf],
             [0, 0, 0]])
        self.eigenvalues = np.array(\
            [ 0.35270742,  2.65176495])
        self.c = random_corpus(1000, 50, 6, 100)
        self.doc_matrix = np.array([0.314334, 0.023485])

    def test_BaseLsaModel_IO(self):
         
        from tempfile import NamedTemporaryFile as NTF
        import os

        tmp = NTF(delete=False, suffix='.npz')      

        try:
            m0 = lsa.BaseLsaModel(context_type='context', 
                word_matrix=self.c.corpus, eigenvalues=self.eigenvalues,
                doc_matrix=self.doc_matrix)
            m0.save(tmp.name)
            m1 = lsa.BaseLsaModel.load(tmp.name)
            
            self.assertEqual(m0.context_type, m1.context_type)
            self.assertTrue((m1.word_matrix == self.c.corpus).all())
            self.assertTrue((m1.eigenvalues == self.eigenvalues).all())
            self.assertTrue((m1.doc_matrix == self.doc_matrix).all())
        
        finally:
            os.remove(tmp.name)
    
    def test_Lsa_train(self):
        m = lsa.Lsa()
        m.train()
        self.assertTrue(m.word_matrix.size == 0)
        self.assertTrue(m.doc_matrix.size == 0)
        self.assertTrue(m.eigenvalues.size == 0)
        
        m = lsa.Lsa(self.tfidf_mat)
        m.train()

        self.assertTrue(np.allclose(self.eigenvalues, m.eigenvalues))


        
#Define and run test suite
suite = unittest.TestLoader().loadTestsFromTestCase(TestLsa)
unittest.TextTestRunner(verbosity=2).run(suite)
