import unittest2 as unittest
import numpy as np

from vsm.corpus.util.corpusbuilders import random_corpus
from vsm.model.base import BaseModel


class TestBaseModel(unittest.TestCase):

    def setUp(self):
        self.c = random_corpus(1000, 50, 6, 100)
        self.m = BaseModel(self.c, 'context')


    def test_BaseModel_IO(self):
           
        from tempfile import NamedTemporaryFile as NTF
        import os

        c = random_corpus(1000, 50, 6, 100)
        tmp = NTF(delete=False, suffix='.npz')
        
        try:
            m0 = BaseModel(c, 'context')
            m0.save(tmp.name)
            m1 = BaseModel.load(tmp.name)

            self.assertEqual(m0.context_type, m1.context_type)
            self.assertTrue((m0.matrix.corpus == m1.matrix.corpus).all())
        finally:
            os.remove(tmp.name)



suite = unittest.TestLoader().loadTestsFromTestCase(TestBaseModel)
unittest.TextTestRunner(verbosity=2).run(suite)
