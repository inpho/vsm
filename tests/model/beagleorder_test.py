import unittest
import os
from tempfile import NamedTemporaryFile
from vsm.util.corpustools import random_corpus
from vsm.model.beagleenvironment import BeagleEnvironment

class TestBeagleOrderModel(unittest.TestCase):

    def test_BeagleOrderSeq(self):
        corp = random_corpus(1000, 50, 0, 20, context_type='sentence')
        env = BeagleEnvironment(corp, n_cols=100)
        env.train()

        model = BeagleOrderSeq(corp, env.matrix)
        model.train()

        try:
            tmp = NamedTemporaryFile(delete=False, suffix='.npz')
            model.save(tmp.name)
            tmp.close()
            loadedModel = BeagleOrderSeq.load(tmp.name)

            modelsEqual = (model.matrix == loadedModel.matrix).all()
        finally:
            os.remove(tmp.name)
        self.assertTrue(modelsEqual, msg=None)

    def test_BeagleOrderMulti(self):
        corp = random_corpus(1000, 50, 0, 20, context_type='sentence')
        env = BeagleEnvironment(corp, n_cols=100)
        env.train()

        model = BeagleOrderMulti(corp, env.matrix)
        model.train(4)

        try:
            tmp = NamedTemporaryFile(delete=False, suffix='.npz')
            model.save(tmp.name)
            tmp.close()
            loadedModel = BeagleOrderSeq.load(tmp.name)

            modelsEqual = (model.matrix == loadedModel.matrix).all()
        finally:
            os.remove(tmp.name)
        self.assertTrue(modelsEqual, msg=None)


if __name__ == '__main__':
    unittest.main()
