import unittest2 as unittest
import numpy as np

from vsm.model.beagleorder import *


class TestBeagleOrder(unittest.TestCase):

    def setUp(self):

        from vsm.corpus.util.corpusbuilders import random_corpus
        from vsm.model.beagleenvironment import BeagleEnvironment

        self.c = random_corpus(1000, 50, 0, 20, context_type='sentence')

        self.e = BeagleEnvironment(self.c, n_cols=100)
        self.e.train()

        self.ms = BeagleOrderSeq(self.c, self.e.matrix)
        self.ms.train()

        self.mm = BeagleOrderMulti(self.c, self.e.matrix)
        self.mm.train(4)


    def test_BeagleOrderSeq(self):
        from tempfile import NamedTemporaryFile
        import os

        try:
            tmp = NamedTemporaryFile(delete=False, suffix='.npz')
            self.ms.save(tmp.name)
            tmp.close()
            m1 = BeagleOrderSeq.load(tmp.name)
            self.assertTrue((self.ms.matrix == m1.matrix).all())
    
        finally:
            os.remove(tmp.name)

        return self.ms.matrix



    def test_BeagleOrderMulti(self):

        from tempfile import NamedTemporaryFile
        import os

        try:
            tmp = NamedTemporaryFile(delete=False, suffix='.npz')
            self.mm.save(tmp.name)
            tmp.close()
            m1 = BeagleOrderMulti.load(tmp.name)
            self.assertTrue((self.mm.matrix == m1.matrix).all())
    
        finally:
            os.remove(tmp.name)

        return self.mm.matrix



    def test_compare(self):

        psi = rand_pt_unit_sphere(self.e.shape[1])

        rand_perm = two_rand_perm(self.e.shape[1])

        print 'Training single processor model'
        ms = BeagleOrderSeq(self.c, self.e.matrix, psi=psi, rand_perm=rand_perm)
        ms.train()

        print 'Training multiprocessor model'
        mm = BeagleOrderMulti(self.c, self.e.matrix, psi=psi, rand_perm=rand_perm)
        mm.train()

        self.assertTrue(np.allclose(ms.matrix, mm.matrix), (ms.matrix, mm.matrix))



    def test10(self):

        import pprint

        def fn(x,y):
            if isinstance(x, tuple):
                return x + (y,)
            return (x, y)

        a = np.arange(5)
        print 'array length', a.shape[0]
        
        for i in xrange(a.shape[0]):
            n = 3
            print 'ngram length', n
            print 'index', i
            pprint.pprint(reduce_ngrams(fn, a, n, i))

        for i in xrange(a.shape[0]):
            n = 4
            print 'ngram length', n
            print 'index', i
            pprint.pprint(reduce_ngrams(fn, a, n, i))

        for i in xrange(a.shape[0]):
            n = 5
            print 'ngram length', n
            print 'index', i
            pprint.pprint(reduce_ngrams(fn, a, n, i))



    def test11(self):

        import pprint

        def fn(x,y):
            return x + y

        a = np.arange(5)
        print 'array length', a.shape[0]
        
        for i in xrange(a.shape[0]):
            n = 3
            print 'ngram length', n
            print 'index', i
            pprint.pprint(reduce_ngrams(fn, a, n, i))

        for i in xrange(a.shape[0]):
            n = 4
            print 'ngram length', n
            print 'index', i
            pprint.pprint(reduce_ngrams(fn, a, n, i))

        for i in xrange(a.shape[0]):
            n = 5
            print 'ngram length', n
            print 'index', i
            pprint.pprint(reduce_ngrams(fn, a, n, i))


        
#Define and run test suite
suite = unittest.TestLoader().loadTestsFromTestCase(TestBeagleOrder)
unittest.TextTestRunner(verbosity=2).run(suite)
