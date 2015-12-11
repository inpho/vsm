"""
Provides the class `LdaCgsViewer`.
"""

import numpy as np

from vsm.spatial import H, JS_dist, KL_div
from vsm.structarr import *
from vsm.split import split_corpus
from vsm.exceptions import *
from types import *
from labeleddata import *
from wrappers import *


__all__ = [ 'LdaCgsViewer' ]


class LdaCgsViewer(object):
    """A class for viewing a topic model estimated by one of vsm's LDA
    classes using CGS.
    """
    
    def __init__(self, corpus, model):
        """
        Initialize LdaCgsViewer.

        :param corpus: Source of observed data.
        :type corpus: :class:`Corpus`
    
        :param model: An LDA model estimated by a CGS.
        :type model: LdaCgsSeq
        """
        self.corpus = corpus
        self.model = model
        self._phi = None
        self._theta = None
        self._H_phi = None
        self._H_theta = None
        self._labels = None


    @property
    def _doc_label_name(self):
        return doc_label_name(self.model.context_type)


    def _res_doc_type(self, doc):
        return res_doc_type(self.corpus, self.model.context_type, 
                              self._doc_label_name, doc)


    def _res_word_type(self, word):
        return res_word_type(self.corpus, word)

    @property
    def labels(self):
        """Returns the list of document labels."""
        if self._labels is None:
            self._labels = self.corpus.view_metadata(self.model.context_type)
            self._labels = self._labels[self._doc_label_name]

        return self._labels


    @property
    def phi(self):
        """Returns the word by topic matrix from the model as a right
        stochastic matrix (the columns phi_i are probability
        distributions).

        """
        if self._phi is None:
            self._phi = self.model.word_top / self.model.word_top.sum(0)
        return self._phi


    @property
    def theta(self):
        """Returns the topic by document matrix from the model as a right
        stochastic matrix (the columns theta_i are probability
        distributions.

        """
        if self._theta is None:
            self._theta =  self.model.top_doc / self.model.top_doc.sum(0)
        return self._theta


    @property
    def H_phi(self):
        """Returns the entropies of the columns of phi (i.e., topics)

        """
        if self._H_phi is None:
            self._H_phi = H(self.phi.T)
        return self._H_phi


    @property
    def H_theta(self):
        """Returns the entropies of the columns of theta.

        """
        if self._H_theta is None:
            self._H_theta = H(self.theta.T)
        return self._H_theta


    def topic_entropies(self, print_len=10):
        """Returns the entropies of the topics of the model as an array sorted
        by entropy.

        """
        H_phi = self.H_phi
        k_arr = enum_sort(H_phi).view(LabeledColumn)
        k_arr.col_header = 'Topic Entropies'
        k_arr.subcol_headers = ['Index', 'Entropy']
        k_arr.col_len = print_len

        return k_arr[::-1]


    def doc_entropies(self, print_len=10, 
                      label_fn=def_label_fn, as_strings=True):
        """Returns the entropies of the distributions over topics as an
        array sorted by entropy.

        """
        if as_strings:
            md = self.corpus.view_metadata(self.model.context_type)
            docs = label_fn(md)
            d_arr = enum_sort(self.H_theta, indices=docs, field_name='doc')
        else:
            d_arr = enum_sort(self.H_theta)

        d_arr = d_arr.view(LabeledColumn)
        d_arr.col_header = 'Document Entropies'
        d_arr.subcol_headers = ['Document', 'Entropy']
        d_arr.col_len = print_len
        
        return d_arr[::-1]
    
    def topic_oscillations(self, print_len=10, div_fn=KL_div):
        """Returns the oscillation in the divergences of documents
        from each topic k, represented as a categorical distribution
        over topics with mass concentrated at index k. 

        Oscillation is computed as the difference between the maximum
        and the minimum of the divergences.

        Returns an array sorted by descending oscillation.
        """
        topic_indices = np.arange(self.model.K)
            
        pseudo_docs = np.diag(np.ones(self.model.K, dtype='d'))[topic_indices, :]
        rel_entropies = div_fn(pseudo_docs, self.theta)
        oscillations = rel_entropies.max(axis=1) - rel_entropies.min(axis=1)

        k_arr = enum_sort(oscillations).view(LabeledColumn)
        k_arr.col_header = 'Topic Oscillation'
        k_arr.subcol_headers = ['Index', 'Oscillation']
        k_arr.col_len = 10
        return k_arr
    
    def topic_jsds(self, print_len=10):
        """Returns the partial N-way JSD of each topic, where N is the number of
        documents in the model. This measure captures the extent to which an
        individual topic is a reliable signal of a document's overall topic
        distribution.
        
        Returns an array sorted by descending partial JSD.
        """
        topic_indices = np.arange(self.model.K)

        doc_tops = self.theta.T
        M = np.sum(doc_tops, axis=0) / len(doc_tops)
        pjsd = np.sum(np.array([(D_i * np.log(D_i / M)) / len(doc_tops) 
                                    for D_i in doc_tops]), axis=0)

        k_arr = enum_sort(pjsd).view(LabeledColumn)
        k_arr.col_header = 'Topic Partial JSD'
        k_arr.subcol_headers = ['Index', 'pJSD']
        k_arr.col_len = 10
        return k_arr[::-1]

    def _get_sort_header_topic_indices(self, sort=None, topic_indices=None):
        """ 
        Returns a tuple of (str, seq) consisting of the column header and sort
        order for a given sort method.

        :param sort: Topic sort function.
        :type sort: string, values are "entropy", "oscillation", "index", "jsd",
            "user" (default if topic_indices set), "index" (default)
        
        :param topic_indices: List of indices of topics to be
            displayed. Default is all topics.
        :type topic_indices: list of integers
        """
        if sort == 'entropy':
            th = 'Topics Sorted by Entropy'
            ent_sort = self.topic_entropies()['i']
            if topic_indices is not None:
                ti = set(topic_indices)
                topic_indices = [k for k in ent_sort if k in ti]
            else:
                topic_indices = ent_sort
        elif sort == 'oscillation':
            th = 'Topics Sorted by Oscillation'
            osc_sort = self.topic_oscillations()['i']
            if topic_indices is not None:
                ti = set(topic_indices)
                topic_indices = [k for k in osc_sort if k in ti]
            else:
                topic_indices = osc_sort
        elif sort == 'jsd':
            th = 'Topics Sorted by Partial JSD'
            jsd_sort = self.topic_jsds()['i']
            if topic_indices is not None:
                ti = set(topic_indices)
                topic_indices = [k for k in jsd_sort if k in ti]
            else:
                topic_indices = jsd_sort

        elif topic_indices is not None or sort == 'user':
            sort = 'user'
            th = 'Topics Sorted by User'
        else:
            sort = 'index'
            th = 'Topics Sorted by Index' 
            topic_indices = range(self.model.K)

        return (th, np.array(topic_indices))
    
    def topics(self, topic_indices=None, sort=None, print_len=10, 
               as_strings=True, compact_view=True, topic_labels=None):
        """
        Returns a list of topics estimated by the model. 
        Each topic is represented by a list of words and the corresponding 
        probabilities.
        
        :param topic_indices: List of indices of topics to be
            displayed. Default is all topics.
        :type topic_indices: list of integers
        
        :param sort: Topic sort function.
        :type sort: string, values are "entropy", "oscillation", "index", "jsd",
            "user" (default if topic_indices set), "index" (default)

        :param print_len: Number of words shown for each topic. Default is 10.
        :type print_len: int, optional

        :param as_string: If `True`, each topic displays words rather than its
            integer representation. Default is `True`.
        :type as_string: boolean, optional
 
        :param compact_view: If `True`, topics are simply represented as
            their top `print_len` number of words. Otherwise, topics are
            shown as words and their probabilities. Default is `True`.
        :type compact_view: boolean, optional       
    
        :param topic_labels: List of strings that are names that correspond
            to the topics in `topic_indices`.
        :type topic_labels: list, optional

        :returns: an instance of :class:`DataTable`.
            A structured array of topics.
        """
        if topic_indices is not None\
            and (not hasattr(topic_indices, '__len__') or\
                isinstance(topic_indices, (str, unicode))):
            raise ValueError("Invalid value for topic_indices," + 
                             "must be a list of integers")

        th, topic_indices = self._get_sort_header_topic_indices(
                                     sort,topic_indices=topic_indices)

        phi = self.phi[:,topic_indices]

        if as_strings:
	        k_arr = enum_matrix(phi.T, indices=self.corpus.words,
                                    field_name='word')
        else:
            ind = [self.corpus.words_int[word] for word in self.corpus.words]
            k_arr = enum_matrix(phi.T, indices=ind, field_name='word')
       
        table = []
        for i,k in enumerate(topic_indices):
            if topic_labels is None:
                ch = 'Topic ' + str(k)
            else:
                ch = topic_labels[i]

            col = LabeledColumn(k_arr[i], col_header=ch, col_len=print_len)
            table.append(col)

        schc = ['Topic', 'Words']
        schf = ['Word', 'Prob']
        return DataTable(table, th, compact_view=compact_view,
                     subcolhdr_compact=schc, subcolhdr_full=schf)


    def doc_topics(self, doc_or_docs, compact_view=False,
                   aggregate=False, print_len=10, topic_labels=None):
        """
        Returns the distribution over topics for the given documents.

        :param doc: Specifies the document whose distribution over topics is 
             returned. It can either be the ID number (integer) or the 
             name (string) of the document.
        :type doc: int or string
        
        :param print_len: Number of topics to be printed. Default is 10.
        :type print_len: int, optional
        
        :param compact_view: If `True`, topics are simply represented as
        their top `print_len` number of words. Otherwise, topics are
        shown as words and their probabilities. Default is `False`.
        :type compact_view: boolean, optional       

        :param topic_labels: List of strings that are names that correspond
            to the topics in `topic_indices`.
        :type topic_labels: list, optional

        :returns: an instance of :class:`LabeledColumn` or of :class: `DataTable`.
            An structured array of topics (represented by their
            number) and their corresponding probabilities or a list of
            such arrays.
        """
        if (isstr(doc_or_docs) or isint(doc_or_docs) 
            or isinstance(doc_or_docs, dict)):
            doc, label = self._res_doc_type(doc_or_docs)

            k_arr = enum_sort(self.theta[:, doc]).view(LabeledColumn)
            k_arr.col_header = 'Document: ' + label
            k_arr.subcol_headers = ['Topic', 'Prob']
            k_arr.col_len = print_len
            return k_arr

        docs, labels = zip(*[self._res_doc_type(d) for d in doc_or_docs])

        k_arr = enum_matrix(self.theta.T, indices=range(self.model.K), 
                            field_name='topic')

        th = 'Distributions over Topics'
        
        table = []
        for i in xrange(len(docs)):
            if topic_labels is None: 
                ch = 'Doc: ' + labels[i]
            else:
                ch = topic_labels[i] 
                
            col = LabeledColumn(k_arr[docs[i]], col_header=ch, col_len=print_len)
            table.append(col)

        schc = ['Doc', 'Topics']
        schf = ['Topic', 'Prob']
        return DataTable(table, th, compact_view=compact_view,
                        subcolhdr_compact=schc, subcolhdr_full=schf)


    def aggregate_doc_topics(self, docs, normed_sum=False, print_len=10):
        """Takes a list of documents identifiers and returns the sum of the
        distributions over topics corresponding to these topics,
        normalized to sum to 1. If normed_sum is True, the sum is
        weighted by document lengths, so that documents contribute
        uniformly to the aggregate distribution.

        """
        docs, labels = zip(*[self._res_doc_type(d) for d in docs])
        
        if normed_sum:
            S = self.theta[:, docs].sum(1)
        else:
            S = self.model.top_doc[:, docs].sum(1)

        S = S / S.sum()

        k_arr = enum_sort(S).view(LabeledColumn)
        k_arr.col_header = 'Aggregate Distribution over Topics'
        k_arr.subcol_headers = ['Topic', 'Prob']
        k_arr.col_len = print_len

        return k_arr
    
    def doc_topic_matrix(self, doc_or_docs):
        """
        Returns the distribution over topics for the given documents.

        :param doc: Specifies the document whose distribution over topics is 
             returned. It can either be the ID number (integer) or the 
             name (string) of the document.
        :type doc: int or string

        :returns: an instance of :class:`np.array`.
            An array of topics
        """
        if (isstr(doc_or_docs) or isint(doc_or_docs) 
            or isinstance(doc_or_docs, dict)):
            doc, label = self._res_doc_type(doc_or_docs)
            k_arr = self.theta[:, doc].T
        else:
            docs, labels = zip(*[self._res_doc_type(d) for d in doc_or_docs])
            k_arr = self.theta[:, docs].T

        return k_arr
    
    def view_documents(self, doc_or_docs, as_strings=False):
        """
        Returns the distribution over topics for the given documents.

        :param doc: Specifies the document whose distribution over topics is 
             returned. It can either be the ID number (integer) or the 
             name (string) of the document.
        :type doc: int or string

        :returns: an instance of :class:`np.array`.
            An array of topics
        """
        documents = []
        if (isstr(doc_or_docs) or isint(doc_or_docs) 
            or isinstance(doc_or_docs, dict)):
            doc, label = self._res_doc_type(doc_or_docs)
            docs = list(doc)
        else:
            docs, labels = zip(*[self._res_doc_type(d) for d in doc_or_docs])
        
        all_docs = self.corpus.view_contexts(self.model.context_type,
                        as_strings=as_strings)
        for doc in docs:
            documents.append(all_docs[doc])

        return documents

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

        :returns: an instance of :class:`LabeledColumn`.
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
        Z = split_corpus(self.model.Z, self.model.indices)
        Z_w = [(d, i, t) for d in xrange(len(Z)) 
               for i,t in enumerate(Z[d]) if idx[d][i]]

        # Label data
        if as_strings:
            docs = self.corpus.view_metadata(ct)[self._doc_label_name]
            dt = [('doc', docs.dtype), ('pos',np.int), ('value', np.int)]
            Z_w = [(docs[d], i, t) for (d, i, t) in Z_w]
        else:
            dt = [('i', np.int), ('pos',np.int), ('value', np.int)]

        Z_w = np.array(Z_w, dtype=dt).view(LabeledColumn)
        Z_w.col_header = 'Word: ' + word
        Z_w.subcol_headers = ['Document', 'Pos', 'Topic']

        return Z_w


    @deprecated_meth("dist_top_top")
    def sim_top_top(self, topic_or_topics, weights=[], 
                     dist_fn=JS_dist, order='i', 
                     show_topics=True, print_len=10, filter_nan=True, 
                     as_strings=True, compact_view=True):
        pass      

    def dist_top_top(self, topic_or_topics, weights=[], 
                     dist_fn=JS_dist, order='i', 
                     show_topics=True, print_len=10, filter_nan=True, 
                     as_strings=True, compact_view=True, topic_labels=None):
        """
        Takes a topic or list of topics (by integer index) and returns
        a list of topics sorted by the distances between a given
        topic and every topic.
        
        :param topic_or_topics: Query topic(s) to which distances are calculated.
        :type topic_or_topics: integer or list of integers
        
        :param weights: Specify weights for each topic in `topic_or_topics`. 
            Default uses equal weights (i.e. arithmetic mean)
        :type weights: list of floating point, optional

        :param show_topics: If `True`, topics are represented by their number
            and distribution over words. Otherwise only topic numbers
            are shown. Default is `True`.
        :type show_topics: boolean, optional

        :param print_len: Number of topics to be shown. Default is 10.
        :type print_len: int, optional       

        :param filter_nan: If `True` not a number entries are filtered.
            Default is `True`.
        :type filter_nan: boolean, optional
        
        :param dist_fn: A distance function from functions in vsm.spatial. 
            Default is :meth:`JS_dist`.
        :type dist_fn: string, optional

        :param order: Order of sorting. 'i' for increasing and 'd' for
            decreasing order. Default is 'i'.
        :type order: string, optional

        :param as_strings: If `True`, words of each topic are represented as
            strings. Otherwise they are represented by their integer
            representation. Default is `True`.
        :type as_strings: boolean, optional

        :param compact_view: If `True`, topics are simply represented as
            their top `print_len` number of words. Otherwise, topics are
            shown as words and their probabilities. Default is `True`.
        :type compact_view: boolean, optional       
        
        :param topic_labels: List of strings that are names that correspond
            to the topics in `topic_indices`.
        :type topic_labels: list, optional

        :returns: an instance of :class:`LabeledColumn`.
            A 2-dim array containing topics and their distances to 
            `topic_or_topics`. 
        
        :See Also: :meth:`vsm.viewer.wrapper.dist_top_top`
        """
        Q = self.model.word_top / self.model.word_top.sum(0)

        distances = dist_top_top(Q, topic_or_topics, weights=weights, 
                                 print_len=print_len, filter_nan=filter_nan, 
                                 dist_fn=dist_fn, order=order)

        if show_topics:
            topic_or_topics = [topic_or_topics]
            topic_indices = distances[distances.dtype.names[0]]
        
            k_arr = self.topics(topic_indices=topic_indices, print_len=print_len,
                                as_strings=as_strings, compact_view=compact_view,
                                topic_labels=topic_labels)
            
            k_arr.table_header = 'Sorted by Topic Distance'
            for i in xrange(distances.size):
                k_arr[i].col_header += ' ({0:.3f})'.format(distances[i][1])

            return k_arr

        return distances 

    
    @deprecated_meth("dist_top_doc")
    def sim_top_doc(self, topic_or_topics, weights=[], filter_words=[],
                     print_len=10, as_strings=True, label_fn=def_label_fn, 
                     filter_nan=True, dist_fn=JS_dist, order='i'):
        pass

    def dist_top_doc(self, topic_or_topics, weights=[], filter_words=[],
                     print_len=10, as_strings=True, label_fn=def_label_fn, 
                     filter_nan=True, dist_fn=JS_dist, order='i'):
        """Takes a topic or list of topics (by integer index) and returns
        a list of documents sorted by distance.

        :param topic_or_topics: Query topic(s) relative to which
            distances are computed.
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
            Default is def_label_fn which retrieves the labels from corpus
            metadata.
        :type label_fn: string, optional

        :param filter_nan: If `True` not a number entries are filtered.
            Default is `True`.
        :type filter_nan: boolean, optional
        
        :param dist_fn: A distance function from functions in vsm.spatial. 
            Default is :meth:`JS_dist`.
        :type dist_fn: string, optional

        :param order: Order of sorting. 'i' for increasing and 'd' for
            decreasing order. Default is 'i'.
        :type order: string, optional

        :returns: an instance of :class:`LabeledColumn`.
            A 2-dim array containing documents and their posterior probabilities 
            to `topic_or_topics`. 

        :See Also: :meth:`def_label_fn`, :meth:`vsm.viewer.wrapper.dist_top_doc`

        """
        Q = self.model.top_doc / self.model.top_doc.sum(0)

        d_arr = dist_top_doc(topic_or_topics, Q, self.corpus,   
                               self.model.context_type, weights=weights, 
                               print_len=print_len, as_strings=False, 
                               label_fn=label_fn, filter_nan=filter_nan, 
                               dist_fn=dist_fn)

        topics = res_top_type(topic_or_topics)

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
            d_arr = map_strarr(d_arr, docs, k='i', new_k='doc')

    	return d_arr

   
    @deprecated_meth("dist_word_top")
    def sim_word_top(self, word_or_words, weights=[], filter_nan=True,
                      show_topics=True, print_len=10, as_strings=True, 
                      compact_view=True, dist_fn=JS_dist, order='i'):
        pass
   
    def dist_word_top(self, word_or_words, weights=[], filter_nan=True,
                      show_topics=True, print_len=10, as_strings=True, 
                      compact_view=True, dist_fn=JS_dist, order='i',
                      topic_labels=None):
        """Sorts topics according to their distance to the query
        `word_or_words`.
        
        A pseudo-topic from `word_or_words` as follows. If weights are
        not provided, the word list is represented in the space of
        topics as a topic which assigns equal non-zero probability to
        each word in `words` and 0 to every other word in the
        corpus. Otherwise, each word in `words` is assigned the
        provided weight.
        
        :param word_or_words: word(s) to which distances are calculated
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

        :param compact_view: If `True`, topics are simply represented as
            their top `print_len` number of words. Otherwise, topics are
            shown as words and their probabilities. Default is `True`.
        :type compact_view: boolean, optional       
        
        :param dist_fn: A distance function from functions in vsm.spatial. 
            Default is :meth:`JS_dist`.
        :type dist_fn: string, optional

        :param order: Order of sorting. 'i' for increasing and 'd' for
            decreasing order. Default is 'i'.
        :type order: string, optional
        
        :param topic_labels: List of strings that are names that correspond
            to the topics in `topic_indices`.
        :type topic_labels: list, optional

        :returns: an instance of :class:`LabeledColumn`.
            A structured array of topics sorted by their distances 
            with `word_or_words`.
        
        :See Also: :meth:`vsm.viewer.wrapper.dist_word_top`

        """
        Q = self.model.word_top / self.model.word_top.sum(0)

        distances = dist_word_top(word_or_words, self.corpus, Q,  
                                  weights=weights, print_len=print_len, 
                                  filter_nan=filter_nan,
                                  dist_fn=dist_fn, order=order)

        if show_topics:
            if isstr(word_or_words):
                word_or_words = [word_or_words]

            # Filter based on topic assignments to words (Z values) 
            topic_indices = sum([self.word_topics(w)['value'].tolist() 
                           for w in word_or_words], [])
            topic_indices = [i for i in xrange(distances.size) if distances[i][0] in topic_indices]
            distances = distances[topic_indices]
            topic_indices = distances[distances.dtype.names[0]]

            k_arr = self.topics(topic_indices=topic_indices, print_len=print_len,
                                as_strings=as_strings, compact_view=compact_view,
                                topic_labels=topic_labels)

            # Retrieve topics
            if compact_view:
                k_arr.table_header = 'Sorted by Topic Distance'
                return k_arr

            # Relabel results
            k_arr.table_header = 'Sorted by Topic Distance'
            for i in xrange(distances.size):
                k_arr[i].col_header += ' ({0:.5f})'.format(distances[i][1])

            return k_arr

        return distances


    @deprecated_meth("dist_doc_doc")
    def sim_doc_doc(self, doc_or_docs, 
                     print_len=10, filter_nan=True, 
                     label_fn=def_label_fn, as_strings=True,
                     dist_fn=JS_dist, order='i'):
        pass

    def dist_doc_doc(self, doc_or_docs, 
                     print_len=10, filter_nan=True, 
                     label_fn=def_label_fn, as_strings=True,
                     dist_fn=JS_dist, order='i'):
        """Computes and sorts the distances between a document 
        or list of documents and every document in the topic space.
 
        :param doc_or_docs: Query document(s) relative to which
        distances are computed.
        :type doc_or_docs: string/integer or list of strings/integer.

        :param print_len: Number of words printed by pretty-printing function.
            Default is 10.
        :type print_len: int, optional

        :param filter_nan: If `True` not a number entries are filtered.
            Default is `True`.
        :type filter_nan: boolean, optional

        :param label_fn: A function that defines how documents are represented.
            Default is def_label_fn which retrieves the labels from corpus
            metadata.
        :type label_fn: string, optional

        :param as_strings: If `True`, returns a list of words rather than
            their integer representations. Default is `True`.
        :type as_strings: boolean, optional
        
        :param dist_fn: A distance function from functions in vsm.spatial. 
            Default is :meth:`JS_dist`.
        :type dist_fn: string, optional

        :param order: Order of sorting. 'i' for increasing and 'd' for
            decreasing order. Default is 'i'.
        :type order: string, optional
       
        :returns: an instance of `LabeledColumn`.
            A 2-dim array containing documents and their distances to 
            `doc_or_docs`. 
        
        :See Also: :meth:`vsm.viewer.wrapper.dist_doc_doc`

        """
        Q = self.model.top_doc / self.model.top_doc.sum(0)

        return dist_doc_doc(doc_or_docs, self.corpus, 
                            self.model.context_type, Q,  
                            print_len=print_len, filter_nan=filter_nan, 
                            label_fn=label_fn, as_strings=as_strings,
                            dist_fn=dist_fn, order=order)

    @deprecated_meth("dismat_doc")
    def simmat_docs(self, docs=[], dist_fn=JS_dist):
        pass
    
    def dismat_doc(self, docs=[], dist_fn=JS_dist):
        """
        Calculates the distance matrix for a given list of documents.

        :param docs: A list of documents whose distance matrix is to be computed.
            Default is all the documents in the model.
        :type docs: list, optional
        
        :param dist_fn: A distance function from functions in vsm.spatial. 
            Default is :meth:`JS_dist`.
        :type dist_fn: string, optional

        :returns: an instance of :class:`IndexedSymmArray`.
            n x n matrix containing floats where n is the number of documents.

        :See Also: :meth:`vsm.viewer.wrapper.dismat_documents`
        """
        if len(docs) == 0:
            docs = range(self.model.top_doc.shape[1])

        Q = self.model.top_doc / self.model.top_doc.sum(0)

        dm =  dismat_doc(docs, self.corpus, self.model.context_type,
                           Q, dist_fn=dist_fn)

        return dm


    @deprecated_meth("dismat_top") 
    def simmat_topics(self, topics=[], dist_fn=JS_dist):
        pass

    def dismat_top(self, topics=[], dist_fn=JS_dist):
        """
        Calculates the distance matrix for a given list of topics.

        :param topic_indices: A list of topics whose distance matrix is to be
            computed. Default is all topics in the model.
        :type topic_indices: list, optional
        
        :param dist_fn: A distance function from functions in vsm.spatial. 
            Default is :meth:`JS_dist`.
        :type dist_fn: string, optional

        :returns: an instance of :class:`IndexedSymmArray`.
            n x n matrix containing floats where n is the number of
            topics considered.
        
        :See Also: :meth:`vsm.viewer.wrapper.dismat_top`
        """

        if len(topics) == 0:
            topics = range(self.model.word_top.shape[1])

        Q = self.model.word_top / self.model.word_top.sum(0)

        return dismat_top(topics, Q, dist_fn=dist_fn)

    def dist(self, doc1, doc2, dist_fn=JS_dist):
        """Computes the distance between 2 documents in topic space.
 
        :param doc1: Query document        
        :type doc1: string/integer
 
        :param doc2: Query document        
        :type doc2: string/integer

        :param dist_fn: A distance function from functions in vsm.spatial. 
            Default is :meth:`JS_dist`.
        :type dist_fn: string, optional
       
        :returns: an instance of `LabeledColumn`.
            A 2-dim array containing documents and their distances to 
            `doc_or_docs`. 
        """
        d1, d2 = self.doc_topic_matrix([doc1,doc2])
        return dist_fn(d1,d2)
        
        

    ######################################################################


    #TODO: Move to plotting extension
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
        
        :returns: an instance of matplotlib.pyplot object.
            Contains the log probability plot. 
        """
        import matplotlib.pyplot as plt

        # If range is not specified, include the whole chain.
        if len(range) == 0:
            range = [0, len(self.model.log_probs)]

        x = []
        logp = []
        for i, lp in self.model.log_probs[range[0]:range[1]:step]:
            x.append(i)
            logp.append(lp)

        plt.plot(x,logp)
        plt.xlim(min(x), max(x))
        plt.grid(grid)
        plt.title('log probability / iteration')

        if show:
            plt.show()

        return plt


    # TODO: Move to plotting extension
    def topic_hist(self, topic_indices=None, d_indices=[], show=True):
        """
        Draws a histogram showing the proportion of topics within a set of
        documents specified by d_indices. 

        :param topic_indices: Specifies the topics for which proportions are 
             calculated. Default is all topics.
        :type doc: list of integers, optional
        
        :param d_indices: Specifies the document for which topic proportions
             are calculated. Default is all documents.
        :type d_indices: list of integers, optional

        :param show: shows plot if `True`. Default is `True`.
        :type d_indices: boolean, optional
        
        :returns: an instance of matplotlib.pyplot object.
             Contains the topic proportion histogram.
        """
        import matplotlib.pyplot as plt

        if len(d_indices) == 0:
            i = self.corpus.context_types.index(self.model.context_type)
            d_indices = xrange(len(self.corpus.context_data[i]))

        td_mat = self.model.top_doc[:, d_indices]
        td_mat /= td_mat.sum(0)
        arr = td_mat.sum(1)

        if not topic_indices == None:
            arr = arr[topic_indices]

        l = enum_sort(arr)
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


