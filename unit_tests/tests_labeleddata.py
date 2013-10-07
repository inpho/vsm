import unittest2 as unittest
import numpy as np

from vsm.viewer.labeleddata import *


class TestLabeleddata(unittest.TestCase):

    def setUp(self):

        words = ['row', 'row', 'row', 'your', 'boat', 'gently', 'down', 'the', 
                 'stream', 'merrily', 'merrily', 'merrily', 'merrily', 'life', 
                 'is', 'but', 'a', 'dream']
        values = [np.random.random() for t in words]
        d = [('i', np.array(words).dtype), 
             ('value', np.array(values).dtype)]
        self.v = np.array(zip(words, values), dtype=d)



    def test_LabeledColumn(self):

        arr = self.v.view(LabeledColumn)
        arr.subcol_headers = ['Word', 'Value']
        arr.col_header = 'Song lets make this longer than subcol headers'
        arr.col_len = 10
        arr1 = self.v.view(LabeledColumn)

        self.assertTrue(type(arr.__str__()) == str)
        self.assertTrue(sum(arr.subcol_widths) < arr.col_width)
        self.assertEqual(arr.shape[0], arr1.col_len)
        self.assertFalse(arr1.col_header)
        self.assertFalse(arr1.subcol_headers)


    def test_DataTable(self):

        v = LabeledColumn(self.v)
        v.subcol_widths = [30, 20]
        v.subcol_headers = ['Word', 'Value']
        v.col_len = 10
        t = []
        for i in xrange(5):
            t.append(v.copy())
            t[i].col_header = 'Iteration ' + str(i)
        t = DataTable(t, 'Song')

        self.assertTrue(type(t.__str__()) == str)
        self.assertTrue('Song', t.table_header)


    def test_CompactTable(self):
        
        li = [[('logic', 0.08902691511387155), ('foundations', 0.08902691511387155),
        ('computer', 0.08902691511387155), ('theoretical', 0.059449866903282994)],
        [('calculus', 0.14554670528602476), ('lambda', 0.14554670528602476),
        ('variety', 0.0731354091238234), ('computation', 0.0731354091238234)],
        [('theology', 0.11285794497473327), ('roman', 0.11285794497473327),
        ('catholic', 0.11285794497473327), ('source', 0.05670971364402021)]]

        arr = np.array(li, dtype=[('words', '|S16'), ('values', '<f8')])

        ct = CompactTable(arr, table_header='Compact view', 
                subcol_headers=['Topic', 'Words'], num_words=4)
        
        self.assertTrue(type(ct.__str__() == str))
        self.assertTrue([1 for s in ct.first_cols if s.startswith('Topic')])     
        self.assertEqual(ct.num_words * (8 +2), ct.subcol_widths[1])
    

    def test_IndexedSymmArray(self):

        from vsm.corpus.util.corpusbuilders import random_corpus
        from vsm.model.ldagibbs import LDAGibbs
        from vsm.viewer.ldagibbsviewer import LDAGibbsViewer

        c = random_corpus(50000, 1000, 0, 50)
        m = LDAGibbs(c, 'context', K=20)
        viewer = LDAGibbsViewer(c, m)
        
        li = ['0', '1', '10']
        isa = viewer.simmat_words(li)
        
        self.assertEqual(isa.shape[0], len(li))
        

  
        
#Define and run test suite
suite = unittest.TestLoader().loadTestsFromTestCase(TestLabeleddata)
unittest.TextTestRunner(verbosity=2).run(suite)
