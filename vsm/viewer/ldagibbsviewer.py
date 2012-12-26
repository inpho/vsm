import numpy as np

from vsm import enum_sort
from vsm import viewer as vw
from vsm.viewer import similarity



class LDAGibbsViewer(object):
    """
    """
    def __init__(self, corpus, model):
        
        self.corpus = corpus

        self.model = model

        self._term_norms = None

        self._doc_norms = None



    def topic(self, n, as_strings=False):
        
        t = self.model.phi_t(n)
        
        t = enum_sort(t)

        if as_strings:

            t = [(self.corpus.terms[i], v) for i,v in t]

        return t


    
    def print_topic(self, n, depth=20):

        t = self.topic(n, as_strings=True)

        if not depth:

            depth = self.model.top_word.shape[1]

        for w, v in t[:depth]:

            print '{0:<20}{1:.4f}'.format(w, v)



    def print_topics(self, depth=10):
        
        topics = []
        
        if not depth:

            depth = self.model.top_word.shape[1]

        for j in xrange(self.model.top_word.shape[0]):

            topics.append(self.topic(j, as_strings=True)[:depth])

        print '-' * 70

        for i in xrange(0, len(topics), 2):

            print '{0:<40}{1}'.format('    Topic ' + str(i), 
                                      '    Topic ' + str(i+1))

            print '-' * 70

            rows = zip(topics[i], topics[i+1])

            for row in rows:

                print ('{0:<20}{1:<20.5f}{2:<20}{3:.5f}'
                       .format(row[0][0], row[0][1], 
                               row[1][0], row[1][1]))

            print '-' * 70


    def relevant_topics(self, term, n):
        """
        Return a list of [n] topic numbers that are most relevant to [term]
        """

        phi_term = self.model.phi_w(self.corpus.terms_int[term])

        top_topics = sorted(range(len(phi_term)), key=lambda i: phi_term[i])[-n:]

        top_topics.reverse()

        return top_topics


    def doc_topics(self, doc_query):
        """
        """
        if isinstance(doc_query, dict):
        
            doc_query = self.corpus.meta_int(self.model.tok_name, doc_query)

        t = self.model.theta_d(doc_query)

        t = enum_sort(t)

        return t


    
    def print_doc_topics(self, doc_query, depth=10):
        """
        """
        t = self.doc_topics(doc_query)

        depth = min(depth, len(t))

        tn = self.model.tok_name

        label = self.corpus.get_metadatum(tn, doc_query, tn + '_label')

        h = 'Document: ' + str(label)

        print

        print '{0:<40}'.format(h)

        print '-' * len(h)

        print '{0:<20}{1:<11}'.format('Topic', 'Probability')

        print '-' * 31

        for t, v in t[:depth]:

            print '  {0:<18}{1:<20.4f}'.format(t, v)



    def word_topics(self, word):
        """
        Takes `word` which is either an integer or a string and
        returns a list of triples `(d, i, t)`:

        d, i : int, int
            Term coordinate of an occurrence of `word`; i.e, the `i`th
            term in the `d`th document
        t : int
            Topic assigned to the `d,i`th term by the model
        """
        if isinstance(word, basestring):
            
            word = self.corpus.terms_int[word]

        idx = [(self.model.W[d] == word) for d in xrange(len(self.model.W))]

        Z = self.model.Z

        Z_w = [(d, i, t) 
               for d in xrange(len(Z)) 
               for i,t in enumerate(Z[d])
               if idx[d][i]]

        return Z_w



    def print_word_topics(self, word, metadata=False):
        """
        """
        Z_w = self.word_topics(word)

        dw = 12

        if metadata:

            docs = self.corpus.view_metadata('documents')

            dw = max([len(d) for d in docs]) + 4

        h = 'Term: ' + str(word)

        print

        print h

        print '-' * len(h)

        print '{0:<{dw}}{1:^13}{2:<5}'.format('Document', 'Rel. pos.', 
                                              'Topic', dw=dw)

        print '-' * (dw + 18)

        for d, i, t in Z_w:
            
            if metadata:

                d = docs[d]

            print ' {0:<{dw}}{1:^13}{2:^5}'.format(d, i, t, dw=(dw-1))



    @property
    def term_norms(self):

        if self._term_norms is None:

            self._term_norms = similarity.col_norms(self.model.top_word)            

        return self._term_norms



    @property
    def doc_norms(self):

        if self._doc_norms is None:

            self._doc_norms = similarity.row_norms(self.model.doc_top)

        return self._doc_norms


    
    def similar_terms(self, term, filter_nan=True, rem_masked=True):

        return vw.similar_terms(self.corpus,
                                self.model.top_word.T,
                                term,
                                norms=self.term_norms,
                                filter_nan=filter_nan,
                                rem_masked=rem_masked)



    def mean_similar_terms(self, query, filter_nan=True, rem_masked=True):

        return vw.mean_similar_terms(self.corpus,
                                     self.model.top_word.T,
                                     query,
                                     norms=self.term_norms,
                                     filter_nan=filter_nan,
                                     rem_masked=rem_masked)



    def similar_documents(self, doc_query, filter_nan=True):

        return vw.similar_documents(self.corpus,
                                    self.model.doc_top.T,
                                    self.model.tok_name,
                                    doc_query,
                                    norms=self.doc_norms,
                                    filter_nan=filter_nan)



    def simmat_terms(self, term_list):

        return vw.simmat_terms(self.corpus,
                               self.model.top_word.T,
                               term_list)



    def simmat_documents(self, doc_queries):

        return vw.simmat_documents(self.corpus,
                                   self.model.doc_top.T,
                                   self.model.tok_name,
                                   doc_queries)
