import unittest2 as unittest

from vsm.corpus import util
from vsm.corpus.util.corpusbuilders import *



class TestCorpusUtil(unittest.TestCase):

    def test_corpus_fromlist(self):

        l = [[],['Not','an','empty','document'],[],
             ['Another','non-empty','document'],[]]

        c = util.corpus_fromlist(l, context_type='sent')

        self.assertTrue(c.context_types == ['sent'])
        self.assertTrue((c.context_data[0]['idx'] == [4,7]).all())
        self.assertTrue((c.context_data[0]['sent_label'] ==
                         ['sent_1', 'sent_3']).all())

    def test_toy_corpus(self):
        keats = ('She dwells with Beauty - Beauty that must die;\n\n'
                 'And Joy, whose hand is ever at his lips\n\n'
                 'Bidding adieu; and aching Pleasure nigh,\n\n'
                 'Turning to poison while the bee-mouth sips:\n\n'
                 'Ay, in the very temple of Delight\n\n'
                 'Veil\'d Melancholy has her sovran shrine,\n\n'
                 'Though seen of none save him whose strenuous tongue\n\n'
                 'Can burst Joy\'s grape against his palate fine;\n\n'
                 'His soul shall taste the sadness of her might,\n\n'
                 'And be among her cloudy trophies hung.')

        self.assertTrue(toy_corpus(keats))
        self.assertTrue(toy_corpus(keats, nltk_stop=True))
        self.assertTrue(toy_corpus(keats, stop_freq=1))
        self.assertTrue(toy_corpus(keats, add_stop=['and', 'with']))
        self.assertTrue(toy_corpus(keats, nltk_stop=True,
                      stop_freq=1, add_stop=['ay']))

        import os
        from tempfile import NamedTemporaryFile as NFT

        tmp = NFT(delete=False)
        tmp.write(keats)
        tmp.close()

        c = toy_corpus(tmp.name, is_filename=True,
                       nltk_stop=True, add_stop=['ay'])
    
        self.assertTrue(c)
        os.remove(tmp.name)

        return c


suite = unittest.TestLoader().loadTestsFromTestCase(TestCorpusUtil)
unittest.TextTestRunner(verbosity=2).run(suite)
