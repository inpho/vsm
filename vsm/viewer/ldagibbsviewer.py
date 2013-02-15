import numpy as np
import matplotlib.pyplot as plt

from vsm import (
    enum_sort as _enum_sort_, 
    map_strarr as _map_strarr_,
    isstr as _isstr_)

from vsm.linalg import row_norms as _row_norms_

from labeleddata import (
    IndexedValueArray as _IndexedValueArray_,
    LabeledColumn as _LabeledColumn_,
    DataTable as _DataTable_,
    format_entry as _format_entry_)

from vsm.viewer import (
    res_term_type as _res_term_type_,
    res_doc_type as _res_doc_type_,
    doc_label_name as _doc_label_name_,
    def_label_fn as _def_label_fn_)

from similarity import (
    sim_word_word as _sim_word_word_,
    sim_doc_doc as _sim_doc_doc_,
    sim_top_top as _sim_top_top_,
    sim_top_doc as _sim_top_doc_,
    sim_word_top as _sim_word_top_,
    simmat_terms as _simmat_terms_,
    simmat_documents as _simmat_documents_,
    simmat_topics as _simmat_topics_)



class LDAGibbsViewer(object):
    """
    """
    def __init__(self, corpus, model):
        """
        """
        self.corpus = corpus
        self.model = model
        self._word_norms_ = None
        self._doc_norms_ = None
        self._topic_norms_ = None
        self._word_sums_ = None
        self._doc_sums_ = None
        self._topic_sums_w_ = None
        self._topic_sums_d_ = None


    @property
    def _doc_label_name(self):
        """
        """
        return _doc_label_name_(self.model.tok_name)


    @property
    def _word_norms(self):
        """
        """
        if self._word_norms_ is None:
            self._word_norms_ = _row_norms_(self.model.top_word.T)            

        return self._word_norms_


    @property
    def _doc_norms(self):
        """
        """
        if self._doc_norms_ is None:
            self._doc_norms_ = _row_norms_(self.model.doc_top)

        return self._doc_norms_


    @property
    def _topic_norms(self):
        """
        """
        if self._topic_norms_ is None:
            self._topic_norms_ = _row_norms_(self.model.top_word)

        return self._topic_norms_


    @property
    def _word_sums(self):
        """
        """
        if self._word_sums_ is None:
            self._word_sums_ = self.model.top_word.sum(0)            

        return self._word_sums_


    @property
    def _doc_sums(self):
        """
        """
        if self._doc_sums_ is None:
            self._doc_sums_ = self.model.doc_top.sum(1)

        return self._doc_sums_


    @property
    def _topic_sums_w(self):
        """
        """
        if self._topic_sums_w_ is None:
            self._topic_sums_w_ = self.model.top_word.sum(1)

        return self._topic_sums_w_


    @property
    def _topic_sums_d(self):
        """
        """
        if self._topic_sums_d_ is None:
            self._topic_sums_d_ = self.model.doc_top.sum(0)

        return self._topic_sums_d_


    def _res_doc_type(self, doc):
        """
        """
        return _res_doc_type_(self.corpus, self.model.tok_name, 
                              self._doc_label_name, doc)


    def _res_term_type(self, term):
        """
        """
        return _res_term_type_(self.corpus, term)


    def topics(self, print_len=10, k_indices=[], as_strings=True):
        """
        """
        if len(k_indices) == 0:
            k_indices = np.arange(self.model.top_word.shape[0])

        # Normalize the topic word matrix so that topics are
        # distributions
        phi = (self.model.top_word[k_indices] / 
               self._topic_sums_w[k_indices][:, np.newaxis])
        
        # Index topics
        k_arr = np.apply_along_axis(_enum_sort_, 1, phi)

        # Label data
        if as_strings:
            f = lambda v: _map_strarr_(v, self.corpus.terms, 
                                       k='i', new_k='word')
            k_arr = np.apply_along_axis(f, 1, k_arr)

        table = []
        for i,k in enumerate(k_indices):
            ch = 'Topic ' + str(k)
            sch = ['Word', 'Prob']
            col = _LabeledColumn_(k_arr[i], col_header=ch,
                                  subcol_headers=sch, col_len=print_len)
            table.append(col)

        table = _DataTable_(table, 'Topics Sorted by Index')

        return table


    def topic_entropies(self, print_len=10, k_indices=[], as_strings=True):
        """
        """
        if len(k_indices) == 0:
            k_indices = np.arange(self.model.top_word.shape[0])
        
        # Normalize the document-topic matrix so that documents are
        # distributions
        theta = (self.model.doc_top[:, k_indices] / 
                 self._topic_sums_d[k_indices][np.newaxis, :])

        # Compute entropy values for each topic
        ent = -1 * (theta * np.log2(theta)).sum(0)

        # Sort topics according to entropies
        k_indices = _enum_sort_(ent)['i'][::-1]
        
        # Retrieve topics
        k_arr = self.topics(print_len=print_len, k_indices=k_indices,
                            as_strings=as_strings)
        
        # Label data
        k_arr.table_header = 'Sorted by Entropy'
        for i in xrange(k_indices.size):
            k_arr[i].col_header += ' ({0:.5f})'.format(ent[k_indices[i]])
 
        return k_arr


    def doc_topics(self, doc, print_len=10):
        """
        """
        d, label = self._res_doc_type(doc)

        # Normalize the document-topic matrix so that documents are
        # distributions
        k_arr = self.model.doc_top[d, :] / self._doc_sums[d]

        # Index, sort and label data
        k_arr = _enum_sort_(k_arr).view(_LabeledColumn_)
        k_arr.col_header = 'Document: ' + label
        k_arr.subcol_headers = ['Topic', 'Prob']
        k_arr.col_len = print_len

        return k_arr


    def word_topics(self, word, as_strings=True):
        """
        """
        w, word = self._res_term_type(word)

        # Search for occurrences of a word in the corpus and return a
        # positions and topic assignments for each found
        idx = [(self.model.W[d] == w) for d in xrange(len(self.model.W))]
        Z = self.model.Z
        Z_w = [(d, i, t) for d in xrange(len(Z)) 
               for i,t in enumerate(Z[d]) if idx[d][i]]

        # Label data
        if as_strings:
            tn = self.model.tok_name
            docs = self.corpus.view_metadata(tn)[self._doc_label_name]
            dt = [('doc', docs.dtype), ('pos',np.int), ('value', np.int)]
            Z_w = [(docs[d], i, t) for (d, i, t) in Z_w]
        else:
            dt = [('i', np.int), ('pos',np.int), ('value', np.int)]

        Z_w = np.array(Z_w, dtype=dt).view(_LabeledColumn_)
        Z_w.col_header = 'Word: ' + word
        Z_w.subcol_headers = ['Document', 'Pos', 'Topic']

        return Z_w


    def doc_finder(self, word, topics, as_strings=True):
        """
        Finds documents and positions where `word` appears with the topic assignment(s) 
        equal to any one of `topics`, and returns a list of documents and positions sorted
        by the relevance of each document to `topics`

        NB: Currently this works only with htrc corpus
        """
        
        doc_prob = dict((doc, prob) for (doc, prob) in self.sim_top_doc(topics))

        doc_list = []
        for (doc, pos), top in self.word_topics(word):
            if any(top == topics):
                doc_list.append(((doc, doc_prob[doc]), pos))

        doc_list.sort(key=lambda tup: tup[0][1], reverse=True)

        # labeling data
        if as_strings:
            metadata = htrc_load_metadata()
            doc_list = [((htrc_get_titles(metadata, d)[0], pr), pos)
                        for ((d, pr), pos) in doc_list]     
            dt = [('i', [('doc', doc_list[0][0][0].dtype), ('prob',np.float)]), ('pos', np.int)]
        else:
            dt = [('i', [('doc', np.int), ('prob',np.float)]), ('pos', np.int)]
            
        doc_list = np.array(doc_list, dtype=dt).view(_IndexedValueArray_)
        doc_list.main_header = 'Word: ' + word + ' by Topic(s)' + str(topics)
        doc_list.subheaders = [('Document, Prob', 'Pos')]

        return doc_list



    def sim_top_top(self, topic_or_topics, weights=None, 
                    print_len=10, filter_nan=True):
        """
        """
        return _sim_top_top_(self.model.top_word, topic_or_topics, 
                             norms=self._topic_norms, weights=weights, 
                             print_len=print_len, filter_nan=filter_nan)


    def sim_top_doc(self, topic_or_topics, weights=[],
                    print_len=10, as_strings=True, label_fn=_def_label_fn_, 
                    filter_nan=True):
        """
        """
        return _sim_top_doc_(self.corpus, self.model.doc_top, topic_or_topics, 
                             self.model.tok_name, weights=weights, 
                             norms=self._doc_norms, print_len=print_len,
                             as_strings=as_strings, label_fn=label_fn, 
                             filter_nan=filter_nan)


    def sim_word_top(self, word_or_words, weights=[], filter_nan=True,
                     show_topics=True, print_len=10, as_strings=True):
        """
        """
        sim = _sim_word_top_(self.corpus, self.model.top_word, word_or_words,
                             weights=weights, norms=self._topic_norms, 
                             print_len=print_len, filter_nan=filter_nan)

        if show_topics:
            if _isstr_(word_or_words):
                word_or_words = [word_or_words]

            # Filter based on topic assignments to words (Z values) 
            k_indices = sum([self.word_topics(w)['value'].tolist() 
                           for w in word_or_words], [])
            k_indices = [i for i in xrange(sim.size) if sim[i][0] in k_indices]
            sim = sim[k_indices]
            k_indices = sim[sim.dtype.names[0]]

            # Retrieve topics
            k_arr = self.topics(k_indices=k_indices, print_len=print_len,
                                as_strings=as_strings)

            # Relabel results
            k_arr.table_header = 'Sorted by Word Similarity'
            for i in xrange(sim.size):
                k_arr[i].col_header += ' ({0:.5f})'.format(sim[i][1])

            return k_arr

        return sim


    def sim_word_word(self, word_or_words, weights=None, 
                      filter_nan=True, print_len=10, as_strings=True):
        """
        """
        return _sim_word_word_(self.corpus, self.model.top_word.T, 
                               word_or_words, weights=weights, 
                               norms=self._word_norms, filter_nan=filter_nan, 
                               print_len=print_len, as_strings=True)


    def sim_doc_doc(self, doc_or_docs, print_len=10, filter_nan=True,
                    label_fn=_def_label_fn_, as_strings=True):
        """
        """
        return _sim_doc_doc_(self.corpus, self.model.doc_top.T,
                             self.model.tok_name, doc_or_docs,
                             norms=self._doc_norms, print_len=print_len,
                             filter_nan=filter_nan, 
                             label_fn=label_fn, as_strings=True)
    

    def simmat_words(self, word_list):

        return _simmat_terms_(self.corpus,
                              self.model.top_word.T,
                              word_list)
    

    def simmat_docs(self, docs):

        return _simmat_documents_(self.corpus,
                                  self.model.doc_top.T,
                                  self.model.tok_name,
                                  docs)


    def simmat_topics(self, topics):

        return _simmat_topics_(self.model.top_word, topics)



    def logp_plot(self, range=[], step=1, show=True, grid=True):
        """
        Returns a plot of log probabilities for `range=[start, end]` by `step` 
        as a matplotlib.pyplot object. If `show=True` the plot is also drawn. 
        """

        # If range is not specified, include the whole chain.
        if not(range):
            range = [0, len(self.model.log_prob)]

        x = []
        logp = []
        for i, lp in self.model.log_prob[range[0]:range[1]:step]:
            x.append(i)
            logp.append(lp)

        plt.plot(x,logp)
        plt.xlim(min(x), max(x))
        plt.grid(grid)
        plt.title('log probability / iteration')

        if show:
            plt.show()

        return plt
