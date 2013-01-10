import numpy as np
import math

from vsm import (enum_sort as _enum_sort, 
                 map_strarr as _map_strarr)

from vsm.viewer import (IndexedValueArray as _IndexedValueArray,
                        res_term_type as _res_term_type_,
                        res_doc_type as _res_doc_type_,
                        doc_label_name as _doc_label_name_,
                        similar_terms as _similar_terms_,
                        mean_similar_terms as _mean_similar_terms_,
                        similar_documents as _similar_documents_,
                        simmat_terms as _simmat_terms_,
                        simmat_documents as _simmat_documents_)

from vsm.viewer.similarity import (col_norms as _col_norms,
                                   row_norms as _row_norms)



class LDAGibbsViewer(object):
    """
    """
    def __init__(self, corpus, model):
        
        self.corpus = corpus

        self.model = model

        self._term_norms = None

        self._doc_norms = None



    @property
    def _doc_label_name(self):

        return _doc_label_name_(self.model.tok_name)



    def _res_doc_type(self, doc):

        return _res_doc_type_(self.corpus, self.model.tok_name, 
                              self._doc_label_name, doc)



    def _res_term_type(self, term):

        return _res_term_type_(self.corpus, term)



    def topics(self, n_terms=None, k_indices=[], as_strings=True):

        if len(k_indices) == 0:

            k_indices = xrange(self.model.top_word.shape[0])
        
        k_arr = []

        for k in k_indices:

            t = self.model.phi_t(k)
        
            t = _enum_sort(t)        
        
            if as_strings:

                t = _map_strarr(t, self.corpus.terms, k='i')

            k_arr.append(t)

        k_arr = np.array(k_arr).view(_IndexedValueArray)

        k_arr.subheaders = [('Topic ' + str(k), 'Prob') 
                              for k in k_indices]

        if n_terms:

            return k_arr[:, :n_terms]

        return k_arr



    def sorted_topics(self, n_terms=None, as_strings=True, word=None, entropy=None):

        if word:

            w, word = self._res_term_type(word)

            phi_w = self.model.phi_w(w)

            k_indices = _enum_sort(phi_w)['i']

            k_arr = self.topics(n_terms=n_terms, k_indices=k_indices, 
                                as_strings=as_strings)

            k_arr.main_header = 'Sorted by Word: ' + word

        elif entropy:

            ent = []

            for i in xrange(self.model.doc_top.shape[1]):

                ent.append((i, self.topic_entropy(i)))

            ent.sort(key=lambda tup: tup[1])

            k_indices = [tup[0] for tup in ent]

            k_arr = self.topics(n_terms=n_terms, k_indices=k_indices,
                                as_strings=as_strings)

            k_arr.main_header = 'Sorted by Entropy'

        else:

            k_arr = self.topics(n_terms=n_terms, as_strings=as_strings)

            k_arr.main_header = 'Sorted by Topic Index'
            
        return k_arr
        


    def topic_entropy(self, t):

        ent = 0.0

        for p in self.model.theta_t(t):

            ent += p * math.log(p, 2)

        ent = -ent

        return ent



    def doc_topics(self, doc, n_topics=None):
        """
        """
        d, label = self._res_doc_type(doc)

        t = self.model.theta_d(d)

        t = _enum_sort(t).view(_IndexedValueArray)

        t.main_header = 'Document: ' + str(label)

        t.subheaders = [('Topic', 'Prob')]

        if n_topics:

            return t[:n_topics]

        return t



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

        Z_w = np.array(Z_w, dtype=dt).view(_IndexedValueArray)

        Z_w.main_header = 'Word: ' + word

        Z_w.subheaders = [('Document, Pos', 'Topic')]

        return Z_w



    @property
    def term_norms(self):

        if self._term_norms is None:

            self._term_norms = _col_norms(self.model.top_word)            

        return self._term_norms



    @property
    def doc_norms(self):

        if self._doc_norms is None:

            self._doc_norms = _row_norms(self.model.doc_top)

        return self._doc_norms


    
    def similar_terms(self, term, filter_nan=True, rem_masked=True):

        return _similar_terms_(self.corpus,
                               self.model.top_word.T,
                               term,
                               norms=self.term_norms,
                               filter_nan=filter_nan,
                               rem_masked=rem_masked)
    


    def mean_similar_terms(self, query, filter_nan=True, rem_masked=True):

        return _mean_similar_terms_(self.corpus,
                                    self.model.top_word.T,
                                    query,
                                    norms=self.term_norms,
                                    filter_nan=filter_nan,
                                    rem_masked=rem_masked)



    def similar_documents(self, doc, filter_nan=True):

        return _similar_documents_(self.corpus,
                                   self.model.doc_top.T,
                                   self.model.tok_name,
                                   doc,
                                   norms=self.doc_norms,
                                   filter_nan=filter_nan)
    


    def simmat_terms(self, term_list):

        return _simmat_terms_(self.corpus,
                              self.model.top_word.T,
                              term_list)
    


    def simmat_documents(self, docs):

        return _simmat_documents_(self.corpus,
                                  self.model.doc_top.T,
                                  self.model.tok_name,
                                  docs)
