import unittest2 as unittest
import numpy as np

from vsm.corpus import Corpus
from vsm.corpus.util.corpusbuilders import random_corpus
from vsm.model.ldacgsseq import *
from vsm.model.ldacgsmulti import *
from multiprocessing import Process

class MPTester:
    def test_demo_LdaCgsMulti(self):
        from vsm.model.ldacgsmulti import demo_LdaCgsMulti
        demo_LdaCgsMulti()
    
    def test_LdaCgsMulti_IO(self):
        from tempfile import NamedTemporaryFile
        import os
    
        c = random_corpus(1000, 50, 6, 100)
        tmp = NamedTemporaryFile(delete=False, suffix='.npz')
        try:
            m0 = LdaCgsMulti(c, 'document', K=10)
            m0.train(n_iterations=20)
            m0.save(tmp.name)
            m1 = LdaCgsMulti.load(tmp.name)
            assert m0.context_type == m1.context_type
            assert m0.K == m1.K
            assert (m0.alpha == m1.alpha).all()
            assert (m0.beta == m1.beta).all()
            assert m0.log_probs == m1.log_probs
            for i in xrange(max(len(m0.corpus), len(m1.corpus))):
                assert m0.corpus[i].all() == m1.corpus[i].all()
            assert m0.V == m1.V
            assert m0.iteration == m1.iteration
            for i in xrange(max(len(m0.Z), len(m1.Z))):
                assert m0.Z[i].all() == m1.Z[i].all()
            assert m0.top_doc.all() == m1.top_doc.all()
            assert m0.word_top.all() == m1.word_top.all()
            assert m0.inv_top_sums.all() == m1.inv_top_sums.all()

            assert m0.seeds == m1.seeds
            for s0, s1 in zip(m0._mtrand_states,m1._mtrand_states):
                assert s0[0] == s1[0]
                assert (s0[1] == s1[1]).all()
                assert s0[2:] == s1[2:]

            m0 = LdaCgsMulti(c, 'document', K=10)
            m0.train(n_iterations=20)
            m0.save(tmp.name)
            m1 = LdaCgsMulti.load(tmp.name)
            assert not hasattr(m1, 'log_prob')
        finally:
            os.remove(tmp.name)

    def test_LdaCgsMulti_SeedTypes(self):
        """ Test for issue #74 issues. """

        from tempfile import NamedTemporaryFile
        import os
    
        c = random_corpus(1000, 50, 6, 100)
        tmp = NamedTemporaryFile(delete=False, suffix='.npz')
        try:
            m0 = LdaCgsMulti(c, 'document', K=10)
            m0.train(n_iterations=20)
            m0.save(tmp.name)
            m1 = LdaCgsMulti.load(tmp.name)

            for s0, s1 in zip(m0.seeds, m1.seeds):
                assert type(s0) == type(s1)
            for s0, s1 in zip(m0._mtrand_states,m1._mtrand_states):
                for i in range(5):
                    assert type(s0[i]) == type(s1[i])
        finally:
            os.remove(tmp.name)

    def test_LdaCgsMulti_random_seeds(self):
        from vsm.corpus.util.corpusbuilders import random_corpus

        c = random_corpus(1000, 50, 0, 20, context_type='document',
                            metadata=True)

        m0 = LdaCgsMulti(c, 'document', K=10)
        assert m0.seeds is not None
        orig_seeds = m0.seeds

        m1 = LdaCgsMulti(c, 'document', K=10, seeds=orig_seeds)
        assert m0.seeds == m1.seeds

        m0.train(n_iterations=5, verbose=0)
        m1.train(n_iterations=5, verbose=0)
        assert m0.seeds == orig_seeds
        assert m1.seeds == orig_seeds

        # ref:http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.get_state.html
        for s0, s1 in zip(m0._mtrand_states,m1._mtrand_states):
            assert s0[0] == 'MT19937'
            assert s1[0] == 'MT19937'
            assert (s0[1] == s1[1]).all()
            assert s0[2:] == s1[2:]

        assert m0.context_type == m1.context_type
        assert m0.K == m1.K
        assert (m0.alpha == m1.alpha).all()
        assert (m0.beta == m1.beta).all()
        assert m0.log_probs == m1.log_probs
        for i in xrange(max(len(m0.corpus), len(m1.corpus))):
            assert m0.corpus[i].all() == m1.corpus[i].all()
        assert m0.V == m1.V
        assert m0.iteration == m1.iteration
        for i in xrange(max(len(m0.Z), len(m1.Z))):
            assert m0.Z[i].all() == m1.Z[i].all()
        assert m0.top_doc.all() == m1.top_doc.all()
        assert m0.word_top.all() == m1.word_top.all()
        assert m0.inv_top_sums.all() == m1.inv_top_sums.all()

    def test_LdaCgsMulti_continue_training(self):
        from vsm.corpus.util.corpusbuilders import random_corpus

        c = random_corpus(1000, 50, 0, 20, context_type='document',
                            metadata=True)

        m0 = LdaCgsMulti(c, 'document', K=10)
        assert m0.seeds is not None
        orig_seeds = m0.seeds

        m1 = LdaCgsMulti(c, 'document', K=10, seeds=orig_seeds)
        assert m0.seeds == m1.seeds

        m0.train(n_iterations=2, verbose=0)
        m1.train(n_iterations=5, verbose=0)
        assert m0.seeds == orig_seeds
        assert m1.seeds == orig_seeds
        for s0, s1 in zip(m0._mtrand_states,m1._mtrand_states):
            assert (s0[1] != s1[1]).any()
            assert s0[2:] != s1[2:]

        m0.train(n_iterations=3, verbose=0)
        # ref:http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.get_state.html
        for s0, s1 in zip(m0._mtrand_states,m1._mtrand_states):
            assert s0[0] == 'MT19937'
            assert s1[0] == 'MT19937'
            assert (s0[1] == s1[1]).all()
            assert s0[2:] == s1[2:]

        assert m0.context_type == m1.context_type
        assert m0.K == m1.K
        assert (m0.alpha == m1.alpha).all()
        assert (m0.beta == m1.beta).all()
        assert m0.log_probs == m1.log_probs
        for i in xrange(max(len(m0.corpus), len(m1.corpus))):
            assert m0.corpus[i].all() == m1.corpus[i].all()
        assert m0.V == m1.V
        assert m0.iteration == m1.iteration
        for i in xrange(max(len(m0.Z), len(m1.Z))):
            assert m0.Z[i].all() == m1.Z[i].all()
        assert m0.top_doc.all() == m1.top_doc.all()
        assert m0.word_top.all() == m1.word_top.all()
        assert m0.inv_top_sums.all() == m1.inv_top_sums.all()
        


    def test_LdaCgsMulti_remove_Seq_props(self):
        from vsm.corpus.util.corpusbuilders import random_corpus

        c = random_corpus(1000, 50, 0, 20, context_type='document',
                            metadata=True)

        m0 = LdaCgsMulti(c, 'document', K=10)

        assert getattr(m0, 'seed', None) is None
        assert getattr(m0, '_mtrand_state', None) is None

    def test_LdaCgsMulti_eq_LdaCgsSeq(self):
        from tempfile import NamedTemporaryFile
        import os
    
        c = random_corpus(1000, 50, 6, 100, seed=2)
        tmp = NamedTemporaryFile(delete=False, suffix='.npz')
        m0 = LdaCgsMulti(c, 'document', K=10, n_proc=1, seeds=[2])
        m1 = LdaCgsSeq(c, 'document', K=10, seed=2)
        for iteration in range(20):
            m0.train(n_iterations=1, verbose=0)
            m1.train(n_iterations=1, verbose=0)
            
            assert m0.context_type == m1.context_type
            assert m0.K == m1.K
            assert (m0.alpha == m1.alpha).all()
            assert (m0.beta == m1.beta).all()
            for i in xrange(max(len(m0.corpus), len(m1.corpus))):
                assert m0.corpus[i].all() == m1.corpus[i].all()
            assert m0.V == m1.V
            assert m0.iteration == m1.iteration
            assert (m0.Z[i] == m1.Z[i]).all()
            assert (m0.top_doc == m1.top_doc).all()
            assert (m0.word_top == m1.word_top).all()
            assert (np.isclose(m0.inv_top_sums, m1.inv_top_sums)).all()
    
            assert m0.seeds[0] == m1.seed
            assert m0._mtrand_states[0][0] == m1._mtrand_state[0]
            for s0,s1 in zip(m0._mtrand_states[0][1], m1._mtrand_state[1]):
                assert s0 == s1
            assert m0._mtrand_states[0][2] == m1._mtrand_state[2]
            assert m0._mtrand_states[0][3] == m1._mtrand_state[3]
            assert m0._mtrand_states[0][4] == m1._mtrand_state[4]
            print iteration, m0.log_probs[-1], m1.log_probs[-1] 
            for i in range(iteration):
                assert np.isclose(m0.log_probs[i][1], m1.log_probs[i][1])
    
    def test_LdaCgsMulti_eq_LdaCgsSeq_multi(self):
        from tempfile import NamedTemporaryFile
        import os
    
        c = random_corpus(1000, 50, 6, 100, seed=2)
        tmp = NamedTemporaryFile(delete=False, suffix='.npz')
        m0 = LdaCgsMulti(c, 'document', K=10, n_proc=1, seeds=[2])
        m1 = LdaCgsSeq(c, 'document', K=10, seed=2)
        for iteration in range(20):
            m0.train(n_iterations=2, verbose=0)
            m1.train(n_iterations=2, verbose=0)
            
            assert m0.context_type == m1.context_type
            assert m0.K == m1.K
            assert (m0.alpha == m1.alpha).all()
            assert (m0.beta == m1.beta).all()
            for i in xrange(max(len(m0.corpus), len(m1.corpus))):
                assert m0.corpus[i].all() == m1.corpus[i].all()
            assert m0.V == m1.V
            assert m0.iteration == m1.iteration
            assert (m0.Z[i] == m1.Z[i]).all()
            assert (m0.top_doc == m1.top_doc).all()
            assert (m0.word_top == m1.word_top).all()
            assert (np.isclose(m0.inv_top_sums, m1.inv_top_sums)).all()
    
            assert m0.seeds[0] == m1.seed
            assert m0._mtrand_states[0][0] == m1._mtrand_state[0]
            for s0,s1 in zip(m0._mtrand_states[0][1], m1._mtrand_state[1]):
                assert s0 == s1
            assert m0._mtrand_states[0][2] == m1._mtrand_state[2]
            assert m0._mtrand_states[0][3] == m1._mtrand_state[3]
            assert m0._mtrand_states[0][4] == m1._mtrand_state[4]
            print iteration, m0.log_probs[-1], m1.log_probs[-1] 
            for i in range(iteration):
                assert np.isclose(m0.log_probs[i][1], m1.log_probs[i][1])


class TestLdaCgsMulti(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_demo_LdaCgsMulti(self):
        t = MPTester()
        p = Process(target=t.test_demo_LdaCgsMulti, args=())
        p.start()
        p.join()
    
    def test_LdaCgsMulti_IO(self):
        t = MPTester()
        p = Process(target=t.test_LdaCgsMulti_IO, args=())
        p.start()
        p.join()
    
    def test_LdaCgsMulti_SeedTypes(self):
        t = MPTester()
        p = Process(target=t.test_LdaCgsMulti_SeedTypes, args=())
        p.start()
        p.join()
    
    def test_LdaCgsMulti_random_seeds(self):
        t = MPTester()
        p = Process(target=t.test_LdaCgsMulti_random_seeds, args=())
        p.start()
        p.join()
    
    def test_LdaCgsMulti_remove_Seq_props(self):
        t = MPTester()
        p = Process(target=t.test_LdaCgsMulti_remove_Seq_props, args=())
        p.start()
        p.join()
    
    def test_LdaCgsMulti_continue_training(self):
        t = MPTester()
        p = Process(target=t.test_LdaCgsMulti_continue_training, args=())
        p.start()
        p.join()
    
    def test_LdaCgsMulti_eq_LdaCgsSeq(self):
        t = MPTester()
        p = Process(target=t.test_LdaCgsMulti_eq_LdaCgsSeq, args=())
        p.start()
        p.join()
    
    def test_LdaCgsMulti_eq_LdaCgsSeq_multi(self):
        t = MPTester()
        p = Process(target=t.test_LdaCgsMulti_eq_LdaCgsSeq_multi, args=())
        p.start()
        p.join()
    

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLdaCgsMulti)
    unittest.TextTestRunner(verbosity=2).run(suite)
