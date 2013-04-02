import unittest
import vsm.model.tf
import numpy as np
import os
from vsm.corpus import BaseCorpus
from tempfile import NamedTemporaryFile

class TestTfModel(unittest.TestCase):

    def setUp(self):
        
        corpus = np.array([0, 3, 2, 1, 0, 3, 0, 2, 3, 0, 2, 3, 1, 
                           2, 0, 3, 2, 1, 2, 2], dtype=np.int)
        contextData = np.array([(3,), (5,), (7,), (11,), (11,), 
                                (15,), (18,), (20,)], 
                               dtype=[('idx', '<i8')])
        self.corpus = BaseCorpus(corpus, context_data=[contextData], 
                                 context_types='document')
        self.model = vsm.model.tf.TfModel(self.corpus, 'document')


    def test_SaveLoad(self):
        """
        Train the model, save it to a temporary file and load it back in.
        """
        self.model.train()

        try:
            tmp = NamedTemporaryFile(delete=False, suffix='.npz')
            self.model.save(tmp.name)
            tmp.close()
            loadedModel = vsm.model.tf.TfModel.load(tmp.name)

            modelsEqual = (self.model.matrix.todense() == loadedModel.matrix.todense()).all()

        # Remove the temporary file
        finally:
            os.remove(tmp.name)

        self.assertTrue(modelsEqual, msg=None)

if __name__ == '__main__':
    unittest.main()
