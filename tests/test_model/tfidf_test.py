import unittest
import vsm.model.tf
import vsm.model.tfidf
import numpy as np
import os
from vsm.corpus import BaseCorpus
from tempfile import NamedTemporaryFile

class TestTfIdfModel(unittest.TestCase):

    def setUp(self):
        
        corpus = np.array([0, 3, 2, 1, 0, 3, 0, 2, 3, 0, 2, 3, 1, 2, 0, 3, 
                            2, 1, 2, 2], dtype=np.int)
        contextData = np.array([(3,), (5,), (7,), (11,), (11,), (15,), (18,), 
                                (20,)], dtype=[('idx', '<i8')])
        self.corpus = BaseCorpus(corpus, context_data=[contextData], 
                                 context_types='document')
        self.tfModel = vsm.model.tf.TfModel(self.corpus, 'document')

        # Train the tf model
        self.tfModel.train()

        # Create the TfIdfModel
        self.tfidfModel = vsm.model.tfidf.TfIdfModel(self.tfModel.matrix,
                                                    'document')


    def test_SaveLoad(self):
        """
        Train the model, save it to a temporary file and load it back in.
        """
        self.tfidfModel.train()

        try:
            tmp = NamedTemporaryFile(delete=False, suffix='.npz')
            self.tfidfModel.save(tmp.name)
            tmp.close()
            loadedModel = vsm.model.tfidf.TfIdfModel.load(tmp.name)
            modelsEqual = (self.tfidfModel.matrix.todense() == loadedModel.matrix.todense()).all()

        # Remove the temporary file
        finally:
            os.remove(tmp.name)

        self.assertTrue(modelsEqual, msg=None)

    def test_TermFreqInvDocFreq(self):
        """
        Test to see if calculated tfidf values align with expected values
        """
        self.tfidfModel.train()
        denseMatrix = self.tfidfModel.matrix.todense()

        # Check to see if the words are in the expected order
        wordsEqual = (np.array([0, 3, 2, 1]) == self.corpus.words).all()
        self.assertTrue(wordsEqual, msg=None)

        # Test the tfidf of word 2 in the eighth document (index 2,7)
        self.assertAlmostEqual(0.94000726, denseMatrix[2,7])

        # Test if the fifth document is empty (index 0-3,4) 
        for i in xrange(0,3):
            self.assertAlmostEqual(0., denseMatrix[i,4])

        # Test if the tfidf of word 0 is the same in the first four
        # documents (index 0,0-3)
        for i in xrange(0,3):
            self.assertAlmostEqual(0.47000363, denseMatrix[0,i])


if __name__ == '__main__':
    unittest.main()
