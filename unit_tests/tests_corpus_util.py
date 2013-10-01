import unittest2 as unittest

from vsm.corpus import util
from vsm.corpus.util.corpusbuilders import *
import numpy as np


class TestCorpusUtil(unittest.TestCase):

    def test_empty_corpus(self):
        
        c = empty_corpus()
        self.assertTrue((np.array([]) == c.corpus).all())
        self.assertTrue(['context'] == c.context_types)
        self.assertTrue((np.array([]) == c.view_contexts('context')).all())

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


    def test_dir_tokenize(self):

        chunks = ['foo foo foo\n\nfoo foo',
                 'Foo bar.  Foo bar.', 
                 '',
                'foo\n\nfoo']

        labels = [str(i) for i in xrange(len(chunks))]
        words, context_data = dir_tokenize(chunks, labels)

        self.assertTrue(len(words) == 11)
        self.assertTrue(len(context_data['article']) == 4)
        self.assertTrue(len(context_data['paragraph']) == 6)
        self.assertTrue(len(context_data['sentence']) == 7)
    
        self.assertTrue((context_data['article']['idx'] == [5, 9, 9, 11]).all())
        self.assertTrue((context_data['article']['article_label'] == 
                ['0', '1', '2', '3']).all())
        self.assertTrue((context_data['paragraph']['idx'] == 
                [3, 5, 9, 9, 10, 11]).all)()
        self.assertTrue((context_data['paragraph']['article_label'] == 
                 ['0', '0', '1', '2', '3', '3']).all())
        self.assertTrue((context_data['paragraph']['par_label'] == 
                 ['0', '1', '2', '3', '4', '5']).all())
        self.assertTrue((context_data['sentence']['idx'] == 
                [3, 5, 7, 9, 9, 10, 11]).all())
        self.assertTrue((context_data['sentence']['article_label'] == 
                ['0', '0', '1', '1', '2', '3', '3']).all())
        self.assertTrue((context_data['sentence']['par_label'] == 
                ['0', '1', '2', '2', '3', '4', '5']).all())
        self.assertTrue((context_data['sentence']['sent_label'] == 
                ['0', '1', '2', '3', '4', '5', '6']).all())



    def test_coll_tokenize(self):

        books = [[('foo foo foo.\n\nfoo foo', '1'),
                  ('Foo bar.  Foo bar.', '2')], 
                 [('','3'),
                ('foo.\n\nfoo', '4')]]

        book_names = [str(i) for i in xrange(len(books))]
        words, context_data = coll_tokenize(books, book_names)

        self.assertTrue(len(words) == 11)
        self.assertTrue(len(context_data['book']) == 2)
        self.assertTrue(len(context_data['page']) == 4)
        self.assertTrue(len(context_data['sentence']) == 7)
        self.assertTrue((context_data['book']['idx'] == [9, 11]).all())
        self.assertTrue((context_data['book']['book_label'] == ['0', '1']).all())
        self.assertTrue((context_data['page']['idx'] == [5, 9, 9, 11]).all())
        self.assertTrue((context_data['page']['page_label'] == 
                            ['0', '1', '2', '3']).all())
        self.assertTrue((context_data['page']['book_label'] == 
                            ['0', '0', '1', '1']).all())
        self.assertTrue((context_data['sentence']['idx'] == 
                            [3, 5, 7, 9, 9, 10, 11]).all())
        self.assertTrue((context_data['sentence']['sent_label'] == 
                ['0', '1', '2', '3', '4', '5', '6']).all())
        self.assertTrue((context_data['sentence']['page_label'] == 
               ['0', '0', '1', '1', '2', '3', '3']).all())
        self.assertTrue((context_data['sentence']['book_label'] == 
                ['0', '0', '0', '0', '1', '1', '1']).all())
        self.assertTrue((context_data['page']['file'] ==
                ['1','2','3','4']).all())
        self.assertTrue((context_data['sentence']['file'] ==
                ['1','1','2','2','3','4','4']).all()) 


suite = unittest.TestLoader().loadTestsFromTestCase(TestCorpusUtil)
unittest.TextTestRunner(verbosity=2).run(suite)
