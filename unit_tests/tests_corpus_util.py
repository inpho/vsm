import unittest2 as unittest

from vsm.corpus import util



class TestCorpusUtil(unittest.TestCase):

    def test_corpus_fromlist(self):

        l = [[],['Not','an','empty','document'],[],
             ['Another','non-empty','document'],[]]

        c = util.corpus_fromlist(l, context_type='sent')

        self.assertTrue(c.context_types == ['sent'])
        self.assertTrue((c.context_data[0]['idx'] == [4,7]).all())
        self.assertTrue((c.context_data[0]['sent_label'] ==
                         ['sent_1', 'sent_3']).all())


#Define and run test suite
suite = unittest.TestLoader().loadTestsFromTestCase(TestCorpusUtil)
unittest.TextTestRunner(verbosity=2).run(suite)
