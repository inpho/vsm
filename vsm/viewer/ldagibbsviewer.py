import numpy as np

from vsm import (enum_sort as _enum_sort_, 
                 map_strarr as _map_strarr_)

from vsm.viewer import (IndexedValueArray as _IndexedValueArray_,
                        res_term_type as _res_term_type_,
                        res_doc_type as _res_doc_type_,
                        doc_label_name as _doc_label_name_,
                        similar_documents as _similar_documents_,
                        simmat_terms as _simmat_terms_,
                        simmat_documents as _simmat_documents_,
                        simmat_topics as _simmat_topics_,
                        sim_top_top as _sim_top_top_,
                        sim_word_avg_top as _sim_word_avg_top_,
                        sim_word_avg_word as _sim_word_avg_word_)

from vsm.viewer.similarity import row_norms as _row_norms_



class LDAGibbsViewer(object):
    """
    """
    def __init__(self, corpus, model):
        
        self.corpus = corpus

        self.model = model

        self._word_norms_ = None

        self._doc_norms_ = None

        self._topic_norms_ = None



    @property
    def _doc_label_name(self):

        return _doc_label_name_(self.model.tok_name)



    def _res_doc_type(self, doc):

        return _res_doc_type_(self.corpus, self.model.tok_name, 
                              self._doc_label_name, doc)



    def _res_term_type(self, term):

        return _res_term_type_(self.corpus, term)



    def topics(self, print_len=10, k_indices=[], as_strings=True):

        if len(k_indices) == 0:

            k_indices = xrange(self.model.top_word.shape[0])
        
        k_arr = []

        for k in k_indices:

            t = self.model.phi_t(k)
        
            t = _enum_sort_(t)        
        
            if as_strings:

                t = _map_strarr_(t, self.corpus.terms, k='i')

            k_arr.append(t)

        k_arr = np.array(k_arr).view(_IndexedValueArray_)

        k_arr.subheaders = [('Topic ' + str(k), 'Prob') 
                              for k in k_indices]

        k_arr.str_len = print_len

        return k_arr



    def sorted_topics(self, print_len=10, as_strings=True, word=None):
        """

        """
        if word:

            w, word = self._res_term_type(word)

            phi_w = self.model.phi_w(w)

            k_indices = _enum_sort_(phi_w)['i']

            wt = self.word_topics(w)

            k_indices = [i for i in k_indices if i in wt['value']]

            k_arr = self.topics(print_len=print_len, k_indices=k_indices, 
                                as_strings=as_strings)

            k_arr.main_header = 'Sorted by Word: ' + word

        else:

            k_arr = self.topics(print_len=print_len, as_strings=as_strings)

            k_arr.main_header = 'Sorted by Topic Index'
            
        return k_arr
        


    def doc_topics(self, doc, print_len=10):
        """
        """
        d, label = self._res_doc_type(doc)

        k_arr = self.model.theta_d(d)

        k_arr = _enum_sort_(k_arr).view(_IndexedValueArray_)

        k_arr.main_header = 'Document: ' + label

        k_arr.subheaders = [('Topic', 'Prob')]

        k_arr.str_len = print_len

        return k_arr



    def word_topics(self, word, as_strings=True):
        """
        Takes `word` which is either an integer or a string and
        returns a structured array of pairs `((d, i), t)`:

        d, i : int or string, int
            Term coordinate of an occurrence of `word`; i.e, the `i`th
            term in the `d`th document. If `as_strings` is True, then
            `d` is assigned the document label associated with the
            document index `d`.
        t : int
            Topic assigned to the `d,i`th term by the model
        """
        w, word = self._res_term_type(word)

        idx = [(self.model.W[d] == w) for d in xrange(len(self.model.W))]

        Z = self.model.Z

        Z_w = [((d, i), t) 
               for d in xrange(len(Z)) 
               for i,t in enumerate(Z[d])
               if idx[d][i]]

        if as_strings:

            tn = self.model.tok_name

            docs = self.corpus.view_metadata(tn)[self._doc_label_name]
        
            dt = [('i', [('doc', docs.dtype), ('pos',np.int)]), 
                  ('value', np.int)]

            Z_w = [((docs[d], i), t) for ((d, i), t) in Z_w]

        else:

            dt = [('i', [('doc', np.int), ('pos',np.int)]), ('value', np.int)]

        Z_w = np.array(Z_w, dtype=dt).view(_IndexedValueArray_)

        Z_w.main_header = 'Word: ' + word

        Z_w.subheaders = [('Document, Pos', 'Topic')]

        return Z_w



    @property
    def _word_norms(self):

        if self._word_norms_ is None:

            self._word_norms_ = _row_norms_(self.model.top_word.T)            

        return self._word_norms_



    @property
    def _doc_norms(self):

        if self._doc_norms_ is None:

            self._doc_norms_ = _row_norms_(self.model.doc_top)

        return self._doc_norms_



    @property
    def _topic_norms(self):

        if self._topic_norms_ is None:

            self._topic_norms_ = _row_norms_(self.model.top_word)

        return self._topic_norms_



    def sim_top_top(self, k, filter_nan=True):
        """
        Computes and sorts the cosine values between a given topic `k`
        and every topic.
        """
        return _sim_top_top_(self.model.top_word, k, norms=self._topic_norms, 
                             filter_nan=filter_nan)


    
    def sim_word_avg_top(self, words, weights=None, filter_nan=True):
        """
        Computes and sorts the cosine values between a list of words
        `words` and every topic. If weights are not provided, the word
        list is represented in the space of topics as a topic which
        assigns equal probability to each word in `words` and 0 to
        every other word in the corpus. Otherwise, each word in
        `words` is assigned the provided weight.
        """
        return _sim_word_avg_top_(self.corpus, self.model.top_word.T, 
                                  words, weights=weights, norms=self._topic_norms,
                                  filter_nan=filter_nan)



    def sim_word_word(self, word, norms=None, 
                      filter_nan=True, as_strings=True):
        """
        Computes and sorts the cosine values between `word` and every
        word.
        """
        return self.sim_word_avg_word([word], filter_nan=filter_nan, 
                                      as_strings=as_strings)



    def sim_word_avg_word(self, words, weights=None, 
                          filter_nan=True, as_strings=True):
        """
        Computes and sorts the cosine values between a list of words
        `words` and every word. If weights are provided, the word list
        is represented as the weighted average of the words in the
        list. If weights are not provided, the arithmetic mean is
        used.
        """
        return _sim_word_avg_word_(self.corpus, self.model.top_word.T, 
                                   words, weights=weights, norms=self._word_norms,
                                   filter_nan=filter_nan, as_strings=True)

    def sim_doc_doc(self, doc, filter_nan=True):

        return _similar_documents_(self.corpus,
                                   self.model.doc_top.T,
                                   self.model.tok_name,
                                   doc,
                                   norms=self._doc_norms,
                                   filter_nan=filter_nan)
    


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
