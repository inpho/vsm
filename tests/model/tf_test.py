import unittest
import vsm.model.tf
import os
from vsm.util.corpustools import random_corpus
from tempfile import NamedTemporaryFile

class TestTfModel(unittest.TestCase):

	def setUp(self):
		self.corpus = random_corpus(10000, 100, 0, 100, context_type='document')
		self.model = vsm.model.tf.TfModel(self.corpus, 'document')


	def test_SaveLoad(self):
		"""
		Train the model, save it to a temporary file and load it back in.
		"""
		self.model.train()

		tmp = NamedTemporaryFile(delete=False, suffix='.npz')
		self.model.save(tmp.name)
		tmp.close()
		loadedModel = vsm.model.tf.TfModel.load(tmp.name)

		modelsEqual = (self.model.matrix.todense() == loadedModel.matrix.todens()).all()

		# Remove the temporary file
		os.remove(tmp.name)

		self.assertTrue(modelsEqual, msg=None)

if __name__ == '__main__':
	unittest.main()