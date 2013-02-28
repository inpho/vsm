import numpy as np

from vsm import (
    enum_sort as _enum_sort_, 
    map_strarr as _map_strarr_,
    isstr as _isstr_)

from vsm.linalg import row_norms as _row_norms_

from labeleddata import (
    LabeledColumn as _LabeledColumn_,
    DataTable as _DataTable_,
    format_entry as _format_entry_)

from vsm.viewer import (
    res_word_type as _res_word_type_,
    res_doc_type as _res_doc_type_,
    res_top_type as _res_top_type_,
    doc_label_name as _doc_label_name_,
    def_label_fn as _def_label_fn_)

from similarity import (
    sim_word_word as _sim_word_word_,
    sim_doc_doc as _sim_doc_doc_,
    sim_top_top as _sim_top_top_,
    sim_top_doc as _sim_top_doc_,
    sim_word_top as _sim_word_top_,
    simmat_words as _simmat_words_,
    simmat_documents as _simmat_documents_,
    simmat_topics as _simmat_topics_)



class LDAGibbsViewer(object):
    """
    A class for viewing a topic model estimated by `LDAGibbs`

    Parameters
    ----------
    corpus : Corpus
        Source of observed data
    model : LDAGibbs object
        A topic modeled fitted by `LDAGibbs`

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
        return _doc_label_name_(self.model.context_type)


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
        return _res_doc_type_(self.corpus, self.model.context_type, 
                              self._doc_label_name, doc)


    def _res_word_type(self, word):
        """
        """
        return _res_word_type_(self.corpus, word)


    def topics(self, print_len=10, k_indices=[], as_strings=True):
        """
        Returns a list of topics estimated by `LDAGibbs` sampler. 
        Each topic is represented by a set of words and the corresponding probabilities.
        
        Parameters
        ----------
        print_len : int
            Number of words shown for each topic. If this is i, i top probability words 
            are shown for each topic. Default is 10.
        k_indices : list of integers
            Order of topics. For example, if k_indices = [3, 0, 2], the 4th, 1st and 3rd topics 
            are printed in this order. Default is ascending from 0 to K-1, where K is the 
            number of topics.
        as_string : boolean
            If true, each topic displays words rather than its ID numbers. Default is True.

        
        Returns
        ----------
        table : a DataTable object.
            A structured array of topics.

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
            f = lambda v: _map_strarr_(v, self.corpus.words, 
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


    def topic_entropies(self, print_len=10, as_strings=True):
        """
        Returns a list of topics sorted according to the entropy of each topic.
        The entropy of topic k is calculated by summing P(d|k) * log(P(d|k) over 
        all document d, and is thought to measure how informative a given topic 
        is to select documents. 
        
        Parameters
        ----------
        print_len : int
            Number of words shown for each topic. If this is i, i top probability words 
            are shown for each topic. Default is 10.
        as_string : boolean
            If true, each topic displays words rather than its ID numbers. Default is True.
        
        Returns
        ----------
        k_arr : a DataTable object.
            A structured array of topics sorted by entropy.

        """
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
        Returns distribution P(K|D=d) over topics K for document d. 

        Parameters
        ----------
        doc : int or string
             Specifies the document whose distribution over topics is returned.
             It can either be the ID number (integer) or the name (string) of the document.
        print_len : int
            Number of topics to be listed. If this is i, i top probability topics are shown.
            Default is 10.
        
        Returns
        ----------
        k_arr : a LabeledColumn object
            An array of topics (represented by their number) and the corresponding 
            probabilities.

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
        Searches for every occurance of `word` in the entire corpus and returns 
        a list each row of which contains the name or ID number of document, 
        the relative position in the document, and the assined topic number 
        for each occurance of `word`.
        
        Parameters
        ----------
        word : string 
            The word for which the search is performed.  
        as_strings : boolean 
            If true, returns document names rather than ID numbers. Default is True.

        Returns
        ----------
        Z_w : a LabeledColumn Object
            A structured array consisting of three columns. Each column is a list of:
            (1) name/ID of document containing `word`
            (2) relative position of `word` in the document
            (3) Topic number assigned to the token.

        """
        w, word = self._res_word_type(word)

        # Search for occurrences of a word in the corpus and return a
        # positions and topic assignments for each found
        idx = [(self.model.W[d] == w) for d in xrange(len(self.model.W))]
        Z = self.model.Z
        Z_w = [(d, i, t) for d in xrange(len(Z)) 
               for i,t in enumerate(Z[d]) if idx[d][i]]

        # Label data
        if as_strings:
            tn = self.model.context_type
            docs = self.corpus.view_metadata(tn)[self._doc_label_name]
            dt = [('doc', docs.dtype), ('pos',np.int), ('value', np.int)]
            Z_w = [(docs[d], i, t) for (d, i, t) in Z_w]
        else:
            dt = [('i', np.int), ('pos',np.int), ('value', np.int)]

        Z_w = np.array(Z_w, dtype=dt).view(_LabeledColumn_)
        Z_w.col_header = 'Word: ' + word
        Z_w.subcol_headers = ['Document', 'Pos', 'Topic']

        return Z_w


    def sim_top_top(self, topic_or_topics, weights=None, 
                    print_len=10, filter_nan=True):
        """
        """
        return _sim_top_top_(self.model.top_word, topic_or_topics, 
                             norms=self._topic_norms, weights=weights, 
                             print_len=print_len, filter_nan=filter_nan)


    def sim_top_doc(self, topic_or_topics, weights=[], filter_words=[],
                    print_len=10, as_strings=True, label_fn=_def_label_fn_, 
                    filter_nan=True):
        """
        """
        d_arr = _sim_top_doc_(self.corpus, self.model.doc_top, topic_or_topics, 
                              self.model.context_type, weights=weights, 
                              norms=self._doc_norms, print_len=print_len,
                              as_strings=False, label_fn=label_fn, 
                              filter_nan=filter_nan)
        
        topics = _res_top_type_(topic_or_topics)

        if len(filter_words) > 0:
            white = set()
            for w in filter_words:
                l = self.word_topics(w, as_strings=False)
                d = l['i'][np.in1d(l['value'], topics)]
                white.update(d)
            
            d_arr = d_arr[(np.in1d(d_arr['i'], white))]

        if as_strings:
            md = self.corpus.view_metadata(self.model.context_type)
            docs = label_fn(md)
            d_arr = _map_strarr_(d_arr, docs, k='i', new_k='doc')

        return d_arr


    def sim_word_top(self, word_or_words, weights=[], filter_nan=True,
                     show_topics=True, print_len=10, as_strings=True):
        """
        A wrapper of `sim_word_top` in similarity.py. 

        Intuitively, the function sorts topics according to their 
        "relevance" to the query `word_or_words`.
        
        Technically, it creates a pseudo-topic consisting of `word_or_words`
        and computes the cosine values between that pseudo-topic and every
        topic in the simplex defined by the probability distribution of words
        conditional on topics. 
        
        If weights are not provided, the word list
        is represented in the space of topics as a topic which assigns
        equal non-zero probability to each word in `words` and 0 to every
        other word in the corpus. Otherwise, each word in `words` is
        assigned the provided weight.
        
        Parameters
        ----------
        word_or_words : string or list of string
            Query word(s) to which cosine values are calculated
        weights : list of floating point
            Specify weights for each query word in `word_or_words`. 
            Default uses equal weights.
        filter_nan : boolean
            ?
        show_topics : boolean
            If true, topics are represented by their number and 
            distribution over words. Otherwise, only topic numbers
            are shown. Default is true.            
        print_len : int
            Number of words printed by pretty-pringing function
            Default is 10.
        as_strings : boolean
            If false, words of each topic are represented by 
            their ID number. Default is true.

        Returns
        ----------
        k_arr : a LabeledColumn object
            A structured array of topics sorted by their cosines values 
            with the query word(s).

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
        A wrapper of `sim_word_word` in similarity.py.
        
        Computes and sorts the cosine values between a word or list of
        words and every word based on the topic distributions.
        Hence a pair of words (w1, w2) is similar according to this 
        function if P(w1|k) ~= P(w2|k) for every topic k. 
        
        If weights are provided, the word list is represented as the 
        weighted average of the words in the list. If weights are not 
        provided, the arithmetic mean is used.

        Parameters
        ----------
        word_or_words : string or list of string
            Query word(s) to which cosine values are calculated
        weights : list of floating point
            Specify weights for each query word in `word_or_words`. 
            Default uses equal weights (i.e. arithmetic mean)
        as_strings : boolean
            If true, returns a list of words rather than IDs. 
            Default is true.
        print_len : int
            Number of words printed by pretty-pringing function
            Default is 10.
        filter_nan : boolean
            ?

        Returns
        ----------
        w_arr : a LabeledColumn object
            A 2-dim array containing words and their cosine values to 
            `word_or_words`. 

        """
        return _sim_word_word_(self.corpus, self.model.top_word.T, 
                               word_or_words, weights=weights, 
                               norms=self._word_norms, filter_nan=filter_nan, 
                               print_len=print_len, as_strings=as_strings)


    def sim_doc_doc(self, doc_or_docs, print_len=10, filter_nan=True,
                    label_fn=_def_label_fn_, as_strings=True):
        """
        """
        return _sim_doc_doc_(self.corpus, self.model.doc_top.T,
                             self.model.context_type, doc_or_docs,
                             norms=self._doc_norms, print_len=print_len,
                             filter_nan=filter_nan, 
                             label_fn=label_fn, as_strings=as_strings)
    

    def simmat_words(self, word_list):

        return _simmat_words_(self.corpus,
                              self.model.top_word.T,
                              word_list)
    

    def simmat_docs(self, docs):

        return _simmat_documents_(self.corpus,
                                  self.model.doc_top.T,
                                  self.model.context_type,
                                  docs)


    def simmat_topics(self, topics):

        return _simmat_topics_(self.model.top_word, topics)



    def logp_plot(self, range=[], step=1, show=True, grid=True):
        """
        Returns a plot of log probabilities for the specified range of the MCMC chain
        used to fit a topic model by `LDAGibbs`.
        The function requires matplotlib package. 

        Parameters
        ----------
        range : list of integer
            Specifies the range of the MCMC chain whose log probabilites are to be plotted.
            For example, range = [0, 999] plots log probabilities from the 1st to the 1000th 
            iterations. The length of the list must be exactly two, and the first element must be
            smaller than the second which can not exceed the total length of the MCMC chain.
            Default produces the plot for the entire chain.
        step : int
            Steps by which points are plotted. Default is 1.
        show : boolean
            If it is True, the function actually draws the plot in addition to returning 
            a plot object. Default is True.
        grid : boolean
            Draw a grid. Default is True. 
        
        Returns
        ----------
        plt : a matplotlib.pyplot object
            Contains the log probability plot. 

        """
        import matplotlib.pyplot as plt

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
