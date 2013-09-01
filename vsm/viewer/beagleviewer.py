from vsm.linalg import row_norms as _row_norms_

from similarity import def_sim_fn, def_simmat_fn, def_order
from similarity import (
    sim_word_word as _sim_word_word_,
    simmat_words as _simmat_words_)



class BeagleViewer(object):
    """
    """
    def __init__(self, corpus, model):
        """
        """
        self.corpus = corpus
        self.model = model
        self._word_norms_ = None


    @property
    def _word_norms(self):
        """
        """
        if self._word_norms_ is None:
            self._word_norms_ = _row_norms_(self.model.matrix)            

        return self._word_norms_


    def sim_word_word(self, word_or_words, weights=None, 
                      filter_nan=True, print_len=10, as_strings=True,
                      sim_fn=def_sim_fn, order=def_order):
        """
        """
        return _sim_word_word_(self.corpus, self.model.matrix, 
                               word_or_words, weights=weights, 
                               norms=self._word_norms, filter_nan=filter_nan, 
                               print_len=print_len, as_strings=True,
                               sim_fn=sim_fn, order=order)


    def simmat_words(self, word_list, sim_fn=def_simmat_fn):

        return _simmat_words_(self.corpus, self.model.matrix,
                              word_list, sim_fn=sim_fn)



def test_BeagleViewer():

    from vsm.corpus.util import random_corpus
    from vsm.model.beagleenvironment import BeagleEnvironment
    from vsm.model.beaglecontext import BeagleContextSeq
    from vsm.model.beagleorder import BeagleOrderSeq
    from vsm.model.beaglecomposite import BeagleComposite

    ec = random_corpus(1000, 50, 0, 20, context_type='sentence')
    cc = ec.apply_stoplist(stoplist=[str(i) for i in xrange(0,50,7)])

    e = BeagleEnvironment(ec, n_cols=5)
    e.train()

    cm = BeagleContextSeq(cc, ec, e.matrix)
    cm.train()

    om = BeagleOrderSeq(ec, e.matrix)
    om.train()

    m = BeagleComposite(cc, cm.matrix, ec, om.matrix)
    m.train()

    venv = BeagleViewer(ec, e)
    vctx = BeagleViewer(cc, cm)
    vord = BeagleViewer(ec, om)
    vcom = BeagleViewer(cc, m)

    return (venv, vctx, vord, vcom)
