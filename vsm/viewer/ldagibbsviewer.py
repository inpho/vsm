import numpy as np

from vsm import viewer as vw
from vsm.viewer import similarity



class LDAGibbsViewer(object):
    """
    """
    def __init__(self,
                 corpus=None,
                 doc_top=None,
                 top_word=None,
                 tok_name=None):
        
        self.corpus = corpus

        self.doc_top = doc_top

        self.top_word = top_word

        self.tok_name = tok_name
        
        # self._term_norms = None

        # self._doc_norms = None



    # @property
    # def term_norms(self):

    #     if self._term_norms is None:

    #         self._term_norms = similarity.row_norms(self.term_matrix)            

    #     return self._term_norms



    # @property
    # def doc_norms(self):

    #     if self._doc_norms is None:

    #         self._doc_norms = similarity.row_norms(self.doc_matrix)            

    #     return self._doc_norms


    
    def get_topic(self, n, as_strings=False):
        
        t = self.top_word[n, :].ravel().tolist()
        
        t = t[:self.corpus.terms.shape[0]]
        
        t = list(enumerate(t))
        
        t = np.array(t, dtype=[('i', np.int), ('v', np.float)])
        
        t.sort(order='v')
        
        t = t[::-1].tolist()
        
        if as_strings:

            t = [(self.corpus.terms[i], v) for i,v in t]

        return t


    
    def print_topic(self, n, depth=20):

        t = self.get_topic(n, as_strings=True)

        for w, v in t[:20]:

            print '{0:<20}{1:<20}'.format(w, str(v)[:-4])


    def print_topics(self, depth=10):
        
        topics = []
        
        for j in xrange(self.top_word.shape[0]):

            topics.append(self.get_topic(j, as_strings=True)[:depth])

        print '-' * 79

        for i in xrange(0, len(topics), 2):

            print '{0:<40}{1:<40}'.format('    Topic ' + str(i), 
                                          '    Topic ' + str(i+1))

            print '-' * 79

            rows = zip(topics[i], topics[i+1])

            for row in rows:

                print ('{0:<20}{1:<20}{2:<20}{3:<20}'
                       .format(row[0][0], str(row[0][1])[:-4], 
                               row[1][0], str(row[1][1])[:-4]))

            print '-' * 79



    # def similar_terms(self, term, filter_nan=True, rem_masked=True):

    #     return vw.similar_terms(self.corpus,
    #                             self.top_word,
    #                             term,
    #                             norms=self.term_norms,
    #                             filter_nan=filter_nan,
    #                             rem_masked=rem_masked)



    # def mean_similar_terms(self, query, filter_nan=True, rem_masked=True):

    #     return vw.mean_similar_terms(self.corpus,
    #                                  self.top_word,
    #                                  query,
    #                                  norms=self.term_norms,
    #                                  filter_nan=filter_nan,
    #                                  rem_masked=rem_masked)



    # def similar_documents(self, document, filter_nan=False):

    #     return vw.similar_documents(self.corpus,
    #                                 self.doc_top,
    #                                 document,
    #                                 norms=self.doc_norms,
    #                                 filter_nan=filter_nan)



    # def simmat_terms(self, term_list):

    #     return vw.simmat_terms(self.corpus,
    #                            self.top_word,
    #                            term_list)



    # def simmat_documents(self, document_list):

    #     return vw.simmat_documents(self.corpus,
    #                                self.doc_top,
    #                                document_list)



def test_ldagibbsviewer_iep():

    from vsm import corpus

    from ldagibbs import LDAGibbs

    c = corpus.Corpus.load('iep-freq1-nltk-compressed.npz')

    m = LDAGibbs()

    m.train(corpus=c, tok_name='articles')

    v = LDAGibbsViewer(c, m.doc_top, m.top_word, K=50, itr=500)

    return m, v



def test_ldagibbsviewer_church():

    from vsm import corpus

    from ldagibbs import LDAGibbs

    m = LDAGibbs()

    c = corpus.Corpus.load('church-nltk.npz')

    m.train(corpus=c, tok_name='documents', K=2, itr=500)

    v = LDAGibbsViewer(c, m.doc_top, m.top_word)

    return m, v
