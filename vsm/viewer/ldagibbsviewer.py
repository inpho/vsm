import numpy as np
import matplotlib.pyplot as plt

from vsm import (
    map_strarr as _map_strarr_,
    enum_matrix as _enum_matrix_,
    enum_sort as _enum_sort_, 
    isstr as _isstr_,
    isint as _isint_)

from vsm.linalg import (
    row_norms as _row_norms_,
    row_cosines as _row_cosines_,
    row_cos_mat as _row_cos_mat_,
    posterior as _posterior_,
    row_kld as _row_kld_,
    row_js_mat as _row_js_mat_)

from labeleddata import (
    LabeledColumn as _LabeledColumn_,
    CompactTable as _CompactTable_,
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
    dismat_words as _dismat_words_,
    dismat_documents as _dismat_documents_,
    dismat_topics as _dismat_topics_)

from manifold import Manifold



class LDAGibbsViewer(object):
    """
    A class for viewing a topic model estimated by `LDAGibbs`.

    :param corpus: Source of observed data.
    :type corpus: Corpus
    
    :param model: A topic modeled fitted by `LDAGibbs`
    :type model: LDAGibbs object

    :attributes:
        * **corpus** (Corpus object) - `corpus`
        * **model** (LDAGibbs object) - `model`

    :methods:
        * :doc:`topics` 
            Returns a list of topics estimated by `LDAGibbs`
            sampler. Each topic is represented by a set of words and the
            corresponding probabilities.
        * :doc:`topic_entropies` 
            Returns topics sorted according to the entropy of each topic. 
            The entropy of topic k is calculated by summing P(d|k) * log(P(d|k))
            over all document d, and is thought to measure how informative a
            given topic is to select documents.
        * :doc:`topic_hist`
            Draws a histogram showing the proportions of topics within a 
            selected set of documents. 
        * :doc:`doc_topics`
            Returns distribution P(K|D=d) over topics K for document d.
        * :doc:`word_topics`
            Searches for every occurrence of `word` in the entire corpus and
            returns a list each row of which contains the name or ID number of
            document, the relative position in the document, and the assigned
            topic number for each occurrence of `word`.
        * :doc:`lda_sim_top_top`
            Returns topics sorted by the cosine similarity values between
            topic(s) and every topic.
        * :doc:`lda_sim_top_doc`
            Returns documents sorted according to their relevance to topic(s).
        * :doc:`lda_sim_word_top`
            Returns topics sorted according to their relevance to word(s).
        * :doc:`lda_sim_word_word`
            Returns words sorted by the cosine values between a word or list
            of words and every word based on the topic distributions.
        * :doc:`lda_sim_doc_doc`
            Computes and sorts the cosine similarity values between a
            document or list of documents and every document in the topic space.
        * :doc:`lda_dismat_words`
            Calculates the distance matrix for a given list of words.
        * :doc:`lda_dismat_docs`
            Calculates the distance matrix for a given list of documents.
        * :doc:`lda_dismat_topics`
            Calculates the distance matrix for a given list of topics.
        * :doc:`logp_plot`
            Returns a plot of log probabilities for the specified range of 
            the MCMC chain used to fit a topic model by `LDAGibbs`.
            The function requires matplotlib package.
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


    def topics(self, print_len=10, k_indices=[], as_strings=True, 
                compact_view=True):
        """
        Returns a list of topics estimated by `LDAGibbs` sampler. 
        Each topic is represented by a set of words and the corresponding 
        probabilities.
        
        :param k_indices: Order of topics. For example, if k_indices = [3, 0, 2], 
            the 4th, 1st and 3rd topics are printed in this order. 
            Default is ascending from 0 to K-1, where K is the 
            number of topics.
        :type k_indices: list of integers
        
        :param print_len: Number of words shown for each topic. If this is i,
            i top probability words are shown for each topic. Default is 10.
        :type print_len: int, optional

        :param as_string: If `True`, each topic displays words rather than its
            integer representation. Default is `True`.
        :type as_string: boolean, optional
        
        :returns: table : :class:`DataTable`.
            A structured array of topics.
        """
        if len(k_indices) == 0:
            k_indices = np.arange(self.model.top_word.shape[0])

        # Normalize the topic word matrix so that topics are
        # distributions
        phi = (self.model.top_word[k_indices] / 
               self._topic_sums_w[k_indices][:, np.newaxis])
        
        # Label data
        if as_strings:
	    k_arr = _enum_matrix_(phi, indices=self.corpus.words,
				 field_name='word')

        # without probabilities, just words
        if compact_view:
            sch = ['Topic', 'Words']
            fc = [str(k) for k in k_indices]
            return _CompactTable_(k_arr, table_header='Topics Sorted by Index',
		    	subcol_headers=sch, first_cols=fc, num_words=print_len)
	
        table = []
        for i,k in enumerate(k_indices):
            ch = 'Topic ' + str(k)
            sch = ['Word', 'Prob']
            col = _LabeledColumn_(k_arr[i], col_header=ch,
                                  subcol_headers=sch, col_len=print_len)
            table.append(col)

        table = _DataTable_(table, 'Topics Sorted by Index')


        return table


    def topic_entropies(self, print_len=10, k_indices=[], as_strings=True, 
                        compact_view=True):
        """
        Returns a list of topics sorted according to the entropy of 
        each topic. The entropy of topic k is calculated by summing 
        P(d|k) * log(P(d|k) over all document d, and is thought to 
        measure how informative a given topic is to select documents. 
        
        :type print_len: int, optional
        :param print_len: Number of words shown for each topic. If this is i,
            i top probability words are shown for each topic. Default is 10.

        :param as_string: If `True`, each topic displays words rather than its
            integer representation. Default is `True`.
        :type as_string: boolean, optional
        
        :returns: k_arr : :class:`DataTable`.
            A structured array of topics sorted by entropy.
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
        if compact_view:
            k_arr = self.topics(print_len=print_len, k_indices=k_indices,
                as_strings=as_strings, compact_view=compact_view)
            k_arr.table_header = 'Sorted by Entropy'
            return k_arr

        k_arr = self.topics(print_len=print_len, k_indices=k_indices,
                            as_strings=as_strings, compact_view=compact_view)

        # Label data
        k_arr.table_header = 'Sorted by Entropy'
        for i in xrange(k_indices.size):
            k_arr[i].col_header += ' ({0:.5f})'.format(ent[k_indices[i]])
 
        return k_arr


    def topic_hist(self, k_indices=[], d_indices=[], show=True):
        """
        Draw a histogram showing the proportion of topics within a set of
        documents specified by d_indices. 

        :param k_indices: Specifies the topics for which proportions are 
             calculated.
        :type doc: list of int
        
        :param d_indices: Specifies the document in which topic proportions
             are culculated. 
        :type d_indices: list of int

        :param show: shows plot if true.
        :type d_indices: boolean
        
        :returns: a matplotlib.pyplot object.
             Contains the topic proportion histogram.
        """

        if len(d_indices) == 0:
            d_indices = xrange(len(self.model.W))

        arr = self.model.theta_d(d_indices).sum(axis=0)

        if len(k_indices) != 0:
            arr = arr[k_indices]

        l = _enum_sort_(arr)
        rank, prob = zip(*l)

        y_pos = np.arange(len(rank))
        fig = plt.figure(figsize=(10,10))

        plt.barh(y_pos, list(prob))
        plt.yticks(y_pos, rank)
        plt.xlabel('Frequencies')
        plt.title('Topic Proportions')

        if show:
            plt.show()

        return plt


    def doc_topics(self, doc, print_len=10):
        """
        Returns distribution P(K|D=d) over topics K for document d. 

        :param doc: Specifies the document whose distribution over topics is 
             returned. It can either be the ID number (integer) or the 
             name (string) of the document.
        :type doc: int or string
        
        :param print_len: Number of topics to be listed. If this is i,
            i top probability topics are shown. Default is 10.
        :type print_len: int, optional
        
        :returns: k_arr : :class:`LabeledColumn`.
            An array of topics (represented by their number) and the 
            corresponding probabilities.
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
        Searches for every occurrence of `word` in the entire corpus and returns 
        a list each row of which contains the name or ID number of document, 
        the relative position in the document, and the assigned topic number 
        for each occurrence of `word`.
        
        :param word: The word for which the search is performed.  
        :type word: string

        :param as_strings: If `True`, returns document names rather than 
            ID numbers. Default is `True`.
        :type as_strings: boolean, optional

        :returns: Z_w : :class:`LabeledColumn`.
            A structured array consisting of three columns. Each column 
            is a list of:
            (1) name/ID of document containing `word`
            (2) relative position of `word` in the document
            (3) Topic number assigned to the token.
        """
        w, word = self._res_word_type(word)

        # Search for occurrences of a word in the corpus and return a
        # positions and topic assignments for each found
        ct = self.model.context_type
        contexts = self.corpus.view_contexts(ct)
        idx = [(contexts[d] == w) for d in xrange(len(contexts))]
        Z = self.model.Z
        Z_w = [(d, i, t) for d in xrange(len(Z)) 
               for i,t in enumerate(Z[d]) if idx[d][i]]

        # Label data
        if as_strings:
            docs = self.corpus.view_metadata(ct)[self._doc_label_name]
            dt = [('doc', docs.dtype), ('pos',np.int), ('value', np.int)]
            Z_w = [(docs[d], i, t) for (d, i, t) in Z_w]
        else:
            dt = [('i', np.int), ('pos',np.int), ('value', np.int)]

        Z_w = np.array(Z_w, dtype=dt).view(_LabeledColumn_)
        Z_w.col_header = 'Word: ' + word
        Z_w.subcol_headers = ['Document', 'Pos', 'Topic']

        return Z_w


    def sim_top_top(self, topic_or_topics, weights=None, 
                    print_len=10, filter_nan=True, sim_fn=_row_kld_):
        """
        Takes a topic or list of topics (by integer index) and returns
        a list of topics sorted by the cosine values between a given
        topic and every topic.
        
        :param topic_or_topics: Query topic(s) to which cosine values are calculated.
        :type topic_or_topics: string or list of strings
        
        :param weights: Specify weights for each topic in `topic_or_topics`. 
            Default uses equal weights (i.e. arithmetic mean)
        :type weights: list of floating point, optional

        :param print_len: Number of topics printed by pretty-pringing function
            Default is 10.
        :type print_len: int, optional       

        :param filter_nan: If `True` not a number entries are filtered.
            Default is `True`.
        :type filter_nan: boolean, optional

        :returns: :class:`LabeledColumn`.
            A 2-dim array containing topics and their cosine values to 
            `topic_or_topics`. 
        
        :See Also: :meth:`vsm.viewer.similarity.sim_top_top`
        """
        return _sim_top_top_(self.model.top_word, topic_or_topics, 
                             norms=self._topic_norms, weights=weights, 
                             print_len=print_len, filter_nan=filter_nan, sim_fn=sim_fn)


    def sim_top_doc(self, topic_or_topics, weights=[], filter_words=[],
                    print_len=10, as_strings=True, label_fn=_def_label_fn_, 
                    filter_nan=True, sim_fn=_row_cosines_):
        """
        Takes a topic or list of topics (by integer index) and returns a 
        list of documents sorted by the posterior probabilities of
        documents given the topic.

        :param topic_or_topics: Query topic(s) to which posterior probabilities
            are calculated.
        :type topic_or_topics: string or list of strings
        
        :param weights: Specify weights for each topic in `topic_or_topics`. 
            Default uses equal weights (i.e. arithmetic mean)
        :type weights: list of floating point, optional

        :param filter_words: The topics that include these words are considered.
            If not provided, by default all topics are considered.
        :type filter_words: list of words, optional
 
        :param print_len: Number of documents printed by pretty-pringing function
            Default is 10.
        :type print_len: int, optional       
        
        :param as_strings: If `True`, returns a list of documents as strings rather
            than their integer representations. Default is `True`.
        :type as_strings: boolean, optional

        :param label_fn: A function that defines how documents are represented.
            Default is def_label_fn which retrieves the labels from corpus metadata.
        :type label_fn: string, optional

        :param filter_nan: If `True` not a number entries are filtered.
            Default is `True`.
        :type filter_nan: boolean, optional

        :returns: :class:`LabeledColumn`.
            A 2-dim array containing documents and their posterior probabilities 
            to `topic_or_topics`. 

        :See Also: :meth:`def_label_fn`, :meth:`vsm.viewer.similarity.sim_top_doc`
        """
        d_arr = _sim_top_doc_(self.corpus, self.model.doc_top, topic_or_topics, 
                              self.model.context_type, weights=weights, 
                              norms=self._doc_norms, print_len=print_len,
                              as_strings=False, label_fn=label_fn, 
                              filter_nan=filter_nan, sim_fn=sim_fn)
        
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
                     show_topics=True, print_len=10, as_strings=True, 
                     compact_view=True):
        """
        A wrapper of `sim_word_top` in similarity.py. 

        Intuitively, the function sorts topics according to their 
        "relevance" to the query `word_or_words`.
        
        Technically, it creates a pseudo-topic consisting of 
        `word_or_words` and computes the cosine values between that 
        pseudo-topic and every topic in the simplex defined by the 
        probability distribution of words conditional on topics. 
        
        If weights are not provided, the word list
        is represented in the space of topics as a topic which assigns
        equal non-zero probability to each word in `words` and 0 to 
        every other word in the corpus. Otherwise, each word in `words` 
        is assigned the provided weight.
        
        :param word_or_words: word(s) to which cosine values are calculated
        :type word_or_words: string or list of strings
        
        :param weights: Specify weights for each query word in `word_or_words`. 
            Default uses equal weights.
        :type weights: list of floating point, optional
        
        :param filter_nan: If `True` not a number entries are filtered.
            Default is `True`.
        :type filter_nan: boolean, optional
            
        :param show_topics: If `True`, topics are represented by their number
            and distribution over words. Otherwise, only topic numbers
            are shown. Default is `True`. 
        :type show_topics: boolean, optional

        :param print_len: Number of words printed by pretty-printing function.
            Default is 10.
        :type print_len: int, optional

        :param as_strings: If `True`, words of each topic are represented as
            strings. Otherwise they are represented by their integer 
            representation. Default is `True`.
        :type as_strings: boolean, optional

        :returns: :class:`LabeledColumn`.
            A structured array of topics sorted by their cosines values 
            with `word_or_words`.
        
        :See Also: :meth:`vsm.viewer.similarity.sim_word_top`
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
            if compact_view:
                k_arr = self.topics(print_len=print_len, k_indices=k_indices,
                    as_strings=as_strings, compact_view=compact_view)
                k_arr.table_header = 'Sorted by Word Similarity'
                return k_arr

            k_arr = self.topics(k_indices=k_indices, print_len=print_len,
                                as_strings=as_strings, compact_view=compact_view)

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

        :param word_or_words: Query word(s) to which cosine values are calculated.
        :type word_or_words: string or list of strings
        
        :param weights: Specify weights for each query word in `word_or_words`. 
            Default uses equal weights (i.e. arithmetic mean)
        :type weights: list of floating point, optional
        
        :param as_strings: If `True`, returns a list of words as strings rather
            than their integer representations. Default is `True`.
        :type as_strings: boolean, optional

        :param print_len: Number of words printed by pretty-printing function
            Default is 10.
        :type print_len: int, optional

        :param filter_nan: If `True` not a number entries are filtered.
            Default is `True`.
        :type filter_nan: boolean, optional

        :returns: :class:`LabeledColumn`.
            A 2-dim array containing words and their cosine values to 
            `word_or_words`. 
        
        :See Also: :meth:`vsm.viewer.similarity.sim_word_word`
        """
        return _sim_word_word_(self.corpus, self.model.top_word.T, 
                               word_or_words, weights=weights, 
                               norms=self._word_norms, filter_nan=filter_nan, 
                               print_len=print_len, as_strings=as_strings)


    def sim_doc_doc(self, doc_or_docs, k_indices=[], print_len=10, filter_nan=True,
                    label_fn=_def_label_fn_, as_strings=True, sim_fn=_row_kld_):
        """
        Computes and sorts the cosine similarity values between a document 
        or list of documents and every document in the topic space. 
        
        :param doc_or_docs: Query document(s) to which cosine values
            are calculated
        :type doc_or_docs: string/integer or list of strings/integers
        
        :param k_indices: A list of topics based on which similarity value is
            computed. Default is all the topics in the model.            
        :type k_indices: list of integers, optional
       
        :param print_len: Number of words printed by pretty-printing function.
            Default is 10.
        :type print_len: int, optional

        :param filter_nan: If `True` not a number entries are filtered.
            Default is `True`.
        :type filter_nan: boolean, optional

        :param label_fn: A function that defines how documents are represented.
            Default is def_label_fn which retrieves the labels from corpus metadata.
        :type label_fn: string, optional

        :param as_strings: If `True`, returns a list of words rather than
            their integer representations. Default is `True`.
        :type as_strings: boolean, optional
        
        :returns: `LabeledColumn`.
            A 2-dim array containing documents and their cosine values to 
            `doc_or_docs`. 
        
        :See Also: :meth:`vsm.viewer.similarity.sim_doc_doc`
        """
        
        if len(k_indices) == 0:
            mat = self.model.doc_top.T
        else:
            mat = self.model.doc_top[:,k_indices].T

        return _sim_doc_doc_(self.corpus, mat, self.model.context_type, 
                             doc_or_docs, norms=self._doc_norms, 
                             print_len=print_len, filter_nan=filter_nan, 
                             label_fn=label_fn, as_strings=as_strings,
                             sim_fn=sim_fn)
    

    def dismat_words(self, word_list):
        """
        Calculates the distance matrix for `word_list`.

        :param word_list: A list of words whose distance matrix is to be
            computed.
        :type word_list: list
        
        :returns: :class:`Manifold`.
            contains n x n matrix containing floats where n is the number of words
            in `word_list`.
       
        :See Also: :meth:`vsm.viewer.similarity.dismat_words`
        """

        dm =  _dismat_words_(self.corpus,
                             self.model.top_word.T,
                             word_list)

        return Manifold(dm, dm.labels)



    def dismat_docs(self, docs=[], k_indices=[], sim_fn=_row_js_mat_):
        """
        Calculates the similarity matrix for a given list of documents.

        :param docs: A list of documents whose similarity matrix is to be computed.
            Default is all the documents in the model.
        :type docs: list, optional
        
        :param k_indices: A list of topics based on which similarity matrix is
            computed. Default is all the topics in the model.
        :type k_indices: list

        :returns: :class:`Manifold`.
            contains n x n matrix containing floats where n is the number of 
            documents.

        :See Also: :meth:`vsm.viewer.similarity.dusmat_documents`
        """

        if len(docs) == 0:
            docs = range(len(self.model.W))

        if len(k_indices) == 0:
            mat = self.model.doc_top.T
        else:
            mat = self.model.doc_top[:,k_indices].T

        dm =  _dismat_documents_(self.corpus, mat,
                                 self.model.context_type,
                                 docs, sim_fn=sim_fn)

        return Manifold(dm, dm.labels)



    def dismat_topics(self, k_indices=[], sim_fn=_row_js_mat_):
        """
        Calculates the similarity matrix for a given list of topics.

        :param k_indices: A list of topics whose similarity matrix is to be
            computed. Default is all topics in the model.
        :type k_indices: list, optional

        :returns: :class:`Manifold`.
            contains n x n matrix containing floats where n is the number of topics
            considered.
        
        :See Also: :meth:`vsm.viewer.similarity.dismat_topics`
        """

        if len(k_indices) == 0:
            k_indices = range(self.model.K)

        dm = _dismat_topics_(self.model.top_word, k_indices, sim_fn=sim_fn)

        # convert strings to integer labels
        labels = [int(x) for x in dm.labels]

        return Manifold(dm, labels)



    def logp_plot(self, range=[], step=1, show=True, grid=True):
        """
        Returns a plot of log probabilities for the specified range of 
        the MCMC chain used to fit a topic model by `LDAGibbs`.
        The function requires matplotlib package. 

        :param range: Specifies the range of the MCMC chain whose log probabilites 
            are to be plotted. For example, range = [0, 999] plots log 
            probabilities from the 1st to the 1000th iterations. 
            The length of the list must be exactly two, and the first 
            element must be smaller than the second which can not exceed 
            the total length of the MCMC chain.
            Default produces the plot for the entire chain.
        :type range: list of integers, optional
        
        :param step: Steps by which points are plotted. Default is 1.
        :type step: int, optional

        :param show: If `True`, the function actually draws the plot 
            in addition to returning a plot object. Default is `True`.
        :type show: boolean, optional

        :param grid: If `True` draw a grid. Default is `True`. 
        :type grid: boolean, optional
        
        :returns: a matplotlib.pyplot object.
            Contains the log probability plot. 
        """

        # If range is not specified, include the whole chain.
        if len(range) == 0:
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
