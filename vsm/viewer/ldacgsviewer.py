"""
Provides the class `LdaCgsViewer`.
"""

import numpy as np

from vsm.spatial import JS_div
from vsm.structarr import *
from types import *
from labeleddata import *
from wrappers import *


__all__ = ['LdaCgsViewer']


class LdaCgsViewer(object):
    """
    A class for viewing a topic model estimated by `LDAGibbs`.
    """
    
    def __init__(self, corpus, model):
        """
        Initialize LdaCgsViewer.

        :param corpus: Source of observed data.
        :type corpus: :class:`Corpus`
    
        :param model: A topic modeled fitted by `LDAGibbs`
        :type model: LDAGibbs
        """
        self.corpus = corpus
        self.model = model


    @property
    def _doc_label_name(self):
        """
        """
        return doc_label_name(self.model.context_type)

    def _res_doc_type(self, doc):
        """
        """
        return res_doc_type(self.corpus, self.model.context_type, 
                              self._doc_label_name, doc)

    def _res_word_type(self, word):
        """
        """
        return res_word_type(self.corpus, word)


    def topics(self, print_len=10, k_indices=[], as_strings=True, 
                compact_view=True):
        """
        Returns a list of topics estimated by `LDAGibbs` sampler. 
        Each topic is represented by a set of words and the corresponding 
        probabilities.
        
        :param k_indices: Indices of topics to be displayed. For example,
            if k_indices = [3, 0, 2], the 4th, 1st and 3rd topics are 
            printed in this order. Default is ascending from 0 to K-1, 
            where K is the number of topics.
        :type k_indices: list of integers
        
        :param print_len: Number of words shown for each topic. Default is 10.
        :type print_len: int, optional

        :param as_string: If `True`, each topic displays words rather than its
            integer representation. Default is `True`.
        :type as_string: boolean, optional
 
        :param compact_view: If `True`, topics are simply represented as
            their top `print_len` number of words. Otherwise, topics are
            shown as words and their probabilities. Default is `True`.
        :type compact_view: boolean, optional       
        
        :returns: an instance of :class:`DataTable`.
            A structured array of topics.
        """
        if len(k_indices) == 0:
            k_indices = np.arange(self.model.top_word.shape[0])

        # Normalize the topic word matrix so that topics are
        # distributions
        phi = self.model.word_top[:,k_indices]
        phi /= phi.sum(0)
        
        # Label data
        if as_strings:
	    k_arr = enum_matrix(phi.T, indices=self.corpus.words,
                                  field_name='word')
        else:
            ind = [self.corpus.words_int[word] for word in self.corpus.words]
            k_arr = enum_matrix(phi.T, indices=ind, field_name='word')


        # without probabilities, just words
        if compact_view:
            sch = ['Topic', 'Words']
            fc = [str(k) for k in k_indices]
            return CompactTable(k_arr, table_header='Topics Sorted by Index',
		    	subcol_headers=sch, first_cols=fc, num_words=print_len)
	
        table = []
        for i,k in enumerate(k_indices):
            ch = 'Topic ' + str(k)
            sch = ['Word', 'Prob']
            col = LabeledColumn(k_arr[i], col_header=ch,
                                  subcol_headers=sch, col_len=print_len)
            table.append(col)

        table = DataTable(table, 'Topics Sorted by Index')


        return table


    #TODO: Use spatial.H to compute entropy
    def topic_entropies(self, print_len=10, as_strings=True, compact_view=True):
        """
        Returns a list of topics sorted according to the entropy of 
        each topic. The entropy of topic k is calculated by summing 
        P(d|k) * log(P(d|k) over all document d, and is thought to 
        measure how informative a given topic is to select documents. 
        
        :type print_len: int, optional
        :param print_len: Number of words shown for each topic. Default is 10.

        :param as_string: If `True`, each topic displays words rather than its
            integer representation. Default is `True`.
        :type as_string: boolean, optional
        
        :param compact_view: If `True`, topics are simply represented as
            their top `print_len` number of words. Otherwise, topics are
            shown as words and their probabilities. Default is `True`.
        :type compact_view: boolean, optional 
        
        :returns: an instance of :class:`DataTable`.
            A structured array of topics sorted by entropy.
        """
        if len(k_indices) == 0:
            k_indices = np.arange(self.model.top_word.shape[0])

        # Normalize the document-topic matrix so that documents are
        # distributions
        theta = self.model.top_doc / self.model.top_doc.sum(0)

        # Compute entropy values for each topic
        ent = -1 * (theta.T * np.log2(theta.T)).sum(0)

        # Sort topics according to entropies
        k_indices = enum_sort(ent)['i'][::-1]
        
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
        Draws a histogram showing the proportion of topics within a set of
        documents specified by d_indices. 

        :param k_indices: Specifies the topics for which proportions are 
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
            d_indices = xrange(len(self.model.W))

        arr = self.model.theta_d(d_indices).sum(axis=0)

        if len(k_indices) != 0:
            arr = arr[k_indices]

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


    def doc_topics(self, doc, print_len=10):
        """
        Returns distribution P(K|D=d) over topics K for document d. 

        :param doc: Specifies the document whose distribution over topics is 
             returned. It can either be the ID number (integer) or the 
             name (string) of the document.
        :type doc: int or string
        
        :param print_len: Number of topics to be listed. Default is 10.
        :type print_len: int, optional
        
        :returns: an instance of :class:`LabeledColumn`.
            An array of topics (represented by their number) and the 
            corresponding probabilities.
        """
        d, label = self._res_doc_type(doc)

        # Normalize the document-topic matrix so that documents are
        # distributions
        k_arr = self.model.top_doc[:,d]
        k_arr /= k_arr.sum()

        # Index, sort and label data
        k_arr = enum_sort(k_arr).view(LabeledColumn)
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

        Z_w = np.array(Z_w, dtype=dt).view(LabeledColumn)
        Z_w.col_header = 'Word: ' + word
        Z_w.subcol_headers = ['Document', 'Pos', 'Topic']

        return Z_w


    def dist_top_top(self, topic_or_topics, weights=[], 
                     dist_fn=JS_div, order='i', 
                     show_topics=True, print_len=10, filter_nan=True, 
                     as_strings=True, compact_view=True):
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
            Default is :meth:`JS_div`.
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
            k_indices = distances[distances.dtype.names[0]]

            # Retrieve topics
            if compact_view:
                k_arr = self.topics(print_len=print_len, k_indices=k_indices,
                                    as_strings=as_strings, 
                                    compact_view=compact_view)
                k_arr.table_header = 'Sorted by Topic Distance'
                return k_arr

            k_arr = self.topics(k_indices=k_indices, print_len=print_len,
                                as_strings=as_strings, compact_view=compact_view)

            # Relabel results
            k_arr.table_header = 'Sorted by Topic Distance'
            for i in xrange(distances.size):
                k_arr[i].col_header += ' ({0:.5f})'.format(distances[i][1])

            return k_arr

        return distances 



    def dist_top_doc(self, topic_or_topics, weights=[], filter_words=[],
                     print_len=10, as_strings=True, label_fn=def_label_fn, 
                     filter_nan=True, dist_fn=JS_div, order='i'):
        """
        Takes a topic or list of topics (by integer index) and returns
        a list of documents sorted by the posterior probabilities of
        documents given the topic(s).

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
            Default is def_label_fn which retrieves the labels from corpus
            metadata.
        :type label_fn: string, optional

        :param filter_nan: If `True` not a number entries are filtered.
            Default is `True`.
        :type filter_nan: boolean, optional
        
        :param dist_fn: A distance function from functions in vsm.spatial. 
            Default is :meth:`JS_div`.
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


    def dist_word_top(self, word_or_words, weights=[], filter_nan=True,
                      show_topics=True, print_len=10, as_strings=True, 
                      compact_view=True, dist_fn=JS_div, order='i'):
        """
        Intuitively, the function sorts topics according to their 
        "relevance" to the query `word_or_words`.
        
        Technically, it creates a pseudo-topic consisting of 
        `word_or_words` and computes distances between that 
        pseudo-topic and every topic.
        
        If weights are not provided, the word list is represented in
        the space of topics as a topic which assigns equal non-zero
        probability to each word in `words` and 0 to every other word
        in the corpus. Otherwise, each word in `words` is assigned the
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
            Default is :meth:`JS_div`.
        :type dist_fn: string, optional

        :param order: Order of sorting. 'i' for increasing and 'd' for
            decreasing order. Default is 'i'.
        :type order: string, optional

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
            k_indices = sum([self.word_topics(w)['value'].tolist() 
                           for w in word_or_words], [])
            k_indices = [i for i in xrange(distances.size) if distances[i][0] in k_indices]
            distances = distances[k_indices]
            k_indices = distances[distances.dtype.names[0]]

            # Retrieve topics
            if compact_view:
                k_arr = self.topics(print_len=print_len, k_indices=k_indices,
                    as_strings=as_strings, compact_view=compact_view)
                k_arr.table_header = 'Sorted by Topic Distance'
                return k_arr

            k_arr = self.topics(k_indices=k_indices, print_len=print_len,
                                as_strings=as_strings, compact_view=compact_view)

            # Relabel results
            k_arr.table_header = 'Sorted by Topic Distance'
            for i in xrange(distances.size):
                k_arr[i].col_header += ' ({0:.5f})'.format(distances[i][1])

            return k_arr

        return distances


    def dist_doc_doc(self, doc_or_docs, 
                     print_len=10, filter_nan=True, 
                     label_fn=def_label_fn, as_strings=True,
                     dist_fn=JS_div, order='i'):
        """
        Computes and sorts the distances between a document 
        or list of documents and every document in the topic space. 
        
        :param doc_or_docs: Query document(s) to which distances
            are calculated
        :type doc_or_docs: string/integer or list of strings/integers
        
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
            Default is :meth:`JS_div`.
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
    

    def dismat_doc(self, docs=[], dist_fn=JS_div):
        """
        Calculates the distance matrix for a given list of documents.

        :param docs: A list of documents whose distance matrix is to be computed.
            Default is all the documents in the model.
        :type docs: list, optional
        
        :param dist_fn: A distance function from functions in vsm.spatial. 
            Default is :meth:`JS_div`.
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



    def dismat_top(self, topics=[], dist_fn=JS_div):
        """
        Calculates the distance matrix for a given list of topics.

        :param k_indices: A list of topics whose distance matrix is to be
            computed. Default is all topics in the model.
        :type k_indices: list, optional
        
        :param dist_fn: A distance function from functions in vsm.spatial. 
            Default is :meth:`JS_div`.
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
