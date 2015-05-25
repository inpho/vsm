import unittest2 as unittest
import numpy as np

from vsm.corpus import Corpus
from vsm.corpus.util.corpusbuilders import random_corpus
from vsm.model.ldacgsmulti import *
from vsm.model.ldacgsseq import *
from vsm.model.lda import *
import platform


class TestLda(unittest.TestCase):
    def setUp(self):
        pass

    def test_Lda_LdaCgsSeq(self):
        m=LDA(multiprocessing=False)
        self.assertTrue(isinstance(m, LdaCgsSeq))
    
    def test_Lda_LdaCgsSeq_seed_or_seeds(self):
        m=LDA(multiprocessing=False, seed_or_seeds=2)
        self.assertTrue(isinstance(m, LdaCgsSeq))
        self.assertTrue(m.seed == 2)
        with self.assertRaises(ValueError):
            m=LDA(multiprocessing=False, seed_or_seeds=[2,4])
    
    
    def test_Lda_proper_class(self):
        m=LDA(multiprocessing=True)
        if platform.system() == 'Windows':
            self.assertTrue(isinstance(m,LdaCgsSeq))
        else:
            self.assertTrue(isinstance(m,LdaCgsMulti))

    def test_Lda_LdaCgsMulti_seed_or_seeds(self):
        m=LDA(multiprocessing=True, seed_or_seeds=[2,4], n_proc=2)
        if platform.system() == 'Windows':
            self.assertTrue(isinstance(m,LdaCgsSeq))
            self.assertTrue(m.seed == 2)
        else:
            self.assertTrue(isinstance(m,LdaCgsMulti))
            self.assertTrue(m.seeds == [2,4])

        # test improper numper of seed_or_seeds with multiprocessing
        with self.assertRaises(ValueError):
            m=LDA(multiprocessing=True, seed_or_seeds=[2], n_proc=2)
        

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLdaCgsMulti)
    unittest.TextTestRunner(verbosity=2).run(suite)
