import unittest2 as unittest

from vsm.corpus import add_metadata
from vsm.extensions.corpusbuilders.util import *
import numpy as np

class TestCorpusUtil(unittest.TestCase):
    
    def test_strip_punc(self):
        
        tsent = ['foo-foo',',','3','foo','bars','bar_foo','2to1','.']
        out = strip_punc(tsent)
        self.assertEqual(out, ['foo-foo','3','foo','bars','bar_foo','2to1'])


    def test_rem_num(self):
 
        tsent = ['foo-foo',',','3','foo','bars','2-parts','2-to-1','3words','.']
        out = rem_num(tsent)
        self.assertEqual(out, ['foo-foo',',','3','foo','bars','2-parts','3words','.'])

    def test_rehyph(self):
        
        sent = 'foo foo 3 foo--bars barfoo -- 2to1.'
        out = rehyph(sent)
        self.assertEqual(out, 'foo foo 3 foo - bars barfoo  -  2to1.')

    def test_add_metadata(self):
        
        from vsm.corpus.util.corpusbuilders import random_corpus

        c = random_corpus(1000, 50, 0, 20, context_type='sentence', metadata=True)
        n = c.view_metadata('sentence').size
        meta = ['m_{0}'.format(i) for i in xrange(n)]
        new_c = add_metadata(c, 'sentence', 'new_meta', meta)

        self.assertEqual(new_c.view_metadata('sentence')['new_meta'].tolist(), meta)


    def test_apply_stoplist(self):
        
        from vsm.corpus.util.corpusbuilders import random_corpus, corpus_fromlist

        c = random_corpus(1000, 50, 0, 20, context_type='sentence', metadata=True)
        new_c = apply_stoplist(c, nltk_stop=False, add_stop=['0','1'], freq=0)

        li = [[],['he','said'],['he','said','bar'],['bar','ate'],['I','foo']]
        wc = corpus_fromlist(li, context_type='sentence')
        new_wc = apply_stoplist(wc, nltk_stop=True, freq=1)
        
        self.assertTrue('0' in c.words)
        self.assertTrue('1' in c.words)
        self.assertFalse('0' in new_c.words)
        self.assertFalse('1' in new_c.words)

        self.assertTrue('said' in new_wc.words)
        self.assertTrue('bar' in new_wc.words)
        self.assertFalse('he' in new_wc.words)
        self.assertFalse('foo' in new_wc.words)
        self.assertFalse('ate' in new_wc.words)


    def test_filter_by_suffix(self):

        li = ['a.txt', 'b.json', 'c.txt']
        filtered = filter_by_suffix(li, ['.txt'])
        filtered1 = filter_by_suffix(li, ['.json'])
        filtered2 = filter_by_suffix(li, ['.csv'])

        self.assertEqual(filtered, ['b.json'])
        self.assertEqual(filtered1, ['a.txt','c.txt'])
        self.assertEqual(filtered2, li)


suite = unittest.TestLoader().loadTestsFromTestCase(TestCorpusUtil)
unittest.TextTestRunner(verbosity=2).run(suite)
