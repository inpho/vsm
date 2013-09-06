import numpy as np

from vsm.linalg import row_norms as _row_norms_
from vsm.linalg import (
    row_acos as _row_acos_,
    row_acos_mat as _row_acos_mat_)

from similarity import (
    sim_word_word as _sim_word_word_,
    simmat_words as _simmat_words_)

from plotting import (
    gen_colors as _gen_colors_,
    plot_clusters as _plot_clusters_)


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
                      sim_fn=_row_acos_, order='i'):
        """
        """
        return _sim_word_word_(self.corpus, self.model.matrix, 
                               word_or_words, weights=weights, 
                               norms=self._word_norms, filter_nan=filter_nan, 
                               print_len=print_len, as_strings=True,
                               sim_fn=sim_fn, order=order)


    def simmat_words(self, word_list, sim_fn=_row_acos_mat_):

        return _simmat_words_(self.corpus, self.model.matrix,
                              word_list, sim_fn=sim_fn)


    # This is a quick adaptation of the isomap_docs function from
    # ldagibbsviewer. This should be abstracted and moved to
    # similarity.py or something equivalent.
    def isomap_words(self, words=[], weights=None, thres=.8,
                     n_neighbors=5, scale=True, trim=20):
        """
        """
        from sklearn import manifold
        from math import ceil

        # create a list to be plotted
        word_list = self.sim_word_word(words, weights=weights)

        # cut down the list by the threshold
        labels, size = zip(*[(w,s) for (w,s) in word_list if s < thres])

        # calculate coordinates
        simmat = self.simmat_words(labels)
        simmat = np.clip(simmat, 0, 2)     # cut off values outside [0, 1]
        imap = manifold.Isomap(n_components=2, n_neighbors=n_neighbors)
        pos  = imap.fit(simmat).embedding_

        # set graphic parameters
        # - scale point size
        if scale:
            size = [s**2*150 for s in size] 
        else:
            size = np.ones_like(size) * 50
        # - trim labels
        if trim:
            labels = [lab[:trim] for lab in labels]

        # hack for unidecode issues in matplotlib
        labels = [label.decode('utf-8', 'ignore') for label in labels]
        
        return _plot_clusters_(pos, labels, size=size)




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
