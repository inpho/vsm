import numpy as np

from vsm import viewer as vw
from vsm.viewer import similarity



def enum_sort(a):

    idx = np.arange(a.size)

    a = np.core.records.fromarrays([idx, a], names='i,v')

    a.sort(order='v')
    
    a = a[::-1]

    return a


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



    def doc_topics(self, doc):
        """
        Takes `doc` which is either a document index (integer) or the
        metadata associated with a document (string) and returns the
        topic distribution for `doc`.
        """
        try:

            doc = self.corpus.meta_int('documents', doc)

        except (TypeError, AttributeError):

            pass

        t = self.model.theta_d(doc)

        t = enum_sort(t)

        return t


    
    def print_doc_topics(self, doc, depth=10, disp_field=None):
        """
        """
        t = self.doc_topics(doc)

        depth = min(depth, len(t))

        if disp_field:

            doc = doc[disp_field]

        h = 'Document: ' + str(doc)

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



    # def similar_documents(self, document, filter_nan=False):

    #     return vw.similar_documents(self.corpus,
    #                                 self.model.doc_top,
    #                                 document,
    #                                 norms=self.doc_norms,
    #                                 filter_nan=filter_nan)



    def simmat_terms(self, term_list):

        return vw.simmat_terms(self.corpus,
                               self.model.top_word.T,
                               term_list)



    # def simmat_documents(self, document_list):

    #     return vw.simmat_documents(self.corpus,
    #                                self.doc_top,
    #                                document_list)
