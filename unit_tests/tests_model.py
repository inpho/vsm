import unittest
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
        with NTF(delete=False, suffix='.npz') as tmp:
            m0 = BaseModel(c.corpus, 'context')
            m0.save(tmp.name)
            m1 = BaseModel.load(tmp.name)

            self.assertEqual(m0.context_type, m1.context_type)
            self.assertTrue((m0.matrix == m1.matrix).all())

        os.remove(tmp.name)

suite = unittest.TestLoader().loadTestsFromTestCase(TestBaseModel)
unittest.TextTestRunner(verbosity=2).run(suite)
