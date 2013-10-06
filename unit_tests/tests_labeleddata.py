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

        self.assertTrue(type(arr.__str__()) == str)


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
    

    #def test_IndexedSymmArray(self):
        

        

  
        
#Define and run test suite
suite = unittest.TestLoader().loadTestsFromTestCase(TestLabeleddata)
unittest.TextTestRunner(verbosity=2).run(suite)
