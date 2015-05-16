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

        self.assertTrue(type(arr.__str__()) == unicode)
        self.assertTrue(sum(arr.subcol_widths) <= arr.col_width)
        self.assertEqual(arr.shape[0], arr1.col_len)
        self.assertFalse(arr1.col_header)
        self.assertFalse(arr1.subcol_headers)


    def test_DataTable(self):

        v = LabeledColumn(self.v)
        v.subcol_widths = [30, 20]
        v.col_len = 10
        t = []
        for i in xrange(5):
            t.append(v.copy())
            t[i].col_header = 'Iteration ' + str(i)
        
        schc = ['Topic', 'Word']
        schf = ['Word', 'Value'] 
        t = DataTable(t, 'Song', subcolhdr_compact=schc, subcolhdr_full=schf)

        self.assertTrue(type(t.__str__()) == unicode)
        self.assertTrue('Song', t.table_header)

        t.compact_view = False
        self.assertTrue(type(t.__str__()) == unicode)
        self.assertTrue('Song', t.table_header)



    def test_IndexedSymmArray(self):

        from vsm.corpus.util.corpusbuilders import random_corpus
        from vsm.model.ldacgsseq import LdaCgsSeq
        from vsm.viewer.ldacgsviewer import LdaCgsViewer

        c = random_corpus(50000, 1000, 0, 50)
        m = LdaCgsSeq(c, 'document', K=20)
        viewer = LdaCgsViewer(c, m)
        
        li = ['0', '1', '10']
        isa = viewer.dismat_top(li)
        
        self.assertEqual(isa.shape[0], len(li))
        

  
        
#Define and run test suite
suite = unittest.TestLoader().loadTestsFromTestCase(TestLabeleddata)
unittest.TextTestRunner(verbosity=2).run(suite)
