import numpy as np

from vsm import enum_sort
from vsm import corpus as cps
from vsm.util.corpustools import word_tokenize
from vsm import model

import similarity



class Viewer(object):
    """
    """
    def __init__(self,
                 corpus=None,
                 matrix=None,
                 tok_name=None):

        self.corpus = corpus
        self.matrix = matrix
        self.tok_name = tok_name


    def load_matrix(self, filename):

        self.matrix = model.Model.load_matrix(filename)


    def load_corpus(self, filename):

        self.corpus = cps.Corpus.load(filename)



def simmat_terms(corpus, matrix, term_list):
    """
    """
    indices = [corpus.terms_int[term] for term in term_list]

    simmat = similarity.simmat_rows(matrix, indices)

    simmat.labels = term_list
    
    return simmat



def simmat_documents(corpus, matrix, tok_name, doc_queries):

    for i,doc_query in enumerate(doc_queries):

        if isinstance(doc_query, dict):

            doc_queries[i] = corpus.meta_int(tok_name, doc_query)

    doc_labels = corpus.view_metadata(tok_name)[tok_name + '_label'][doc_queries]
        
    simmat = similarity.simmat_columns(matrix, doc_queries)

    simmat.labels = doc_labels

    return simmat



def similar_terms(corpus, matrix, term,
                  norms=None, filter_nan=True,
                  rem_masked=True):

    i = corpus.terms_int[term]
    
    sim_vals = similarity.similar_rows(i, matrix,
                                       norms=norms,
                                       filter_nan=filter_nan)

    sim_vals = np.array([(corpus.terms[i], v) for i,v in sim_vals],
                        dtype=[('term', corpus.terms.dtype),
                               ('value', sim_vals.dtype['v'])])

    if rem_masked:

        f = np.vectorize(lambda x: x is not np.ma.masked)

        sim_vals = sim_vals[f(sim_vals['term'])]

    sim_vals = sim_vals.view(TermValueArray)

    sim_vals.main_header = term

    sim_vals.subheaders = [('Term', 'Cosine')]

    return sim_vals



def similar_documents(corpus, matrix, tok_name, doc_query,
                      norms=None, filter_nan=True):

    if isinstance(doc_query, dict):
        
        doc_query = corpus.meta_int(tok_name, doc_query)
    
    sim_vals = similarity.similar_columns(doc_query, matrix, norms=norms,
                                          filter_nan=filter_nan)

    docs = corpus.view_metadata(tok_name)[tok_name + '_label']

    sim_vals = np.array([(docs[i], v) for i,v in sim_vals],
                        dtype=[('doc', docs.dtype),
                               ('value', sim_vals.dtype['v'])])

    sim_vals = sim_vals.view(TermValueArray)

    sim_vals.main_header = docs[doc_query]

    sim_vals.subheaders = [('Document', 'Cosine')]

    return sim_vals



def mean_similar_terms(corpus, matrix, query,
                       norms=None, filter_nan=True,
                       rem_masked=True):

    terms = word_tokenize(query)

    def sim_terms(term):

        i = corpus.terms_int[term]
    
        ra = similarity.similar_rows(i, matrix, norms=norms,
                                     sort=False, filter_nan=False)

        return ra['v']



    sim_vals = [sim_terms(term) for term in terms]

    sim_vals = reduce(np.add, sim_vals)
    
    sim_vals = sim_vals / len(terms)

    sim_vals = enum_sort(sim_vals)
    
    if filter_nan:

        sim_vals = sim_vals[np.isfinite(sim_vals['v'])]

    sim_vals = np.array([(corpus.terms[i], v) for i,v in sim_vals],
                        dtype=[('term', corpus.terms.dtype),
                               ('value', sim_vals.dtype['v'])])

    if rem_masked:

        f = np.vectorize(lambda x: x is not np.ma.masked)

        sim_vals = sim_vals[f(sim_vals['term'])]

    sim_vals = sim_vals.view(TermValueArray)

    sim_vals.main_header = query

    sim_vals.subheaders = [('Term', 'Cosine')]

    return sim_vals



class TermValueArray(np.ndarray):
    """
    """
    def __new__(cls, input_array, main_header=None, subheaders=None):

        obj = np.asarray(input_array).view(cls)

        obj.main_header = main_header

        obj.subheaders = subheaders

        return obj



    def __array_finalize__(self, obj):

        if obj is None: return

        self.main_header = getattr(obj, 'main_header', None)

        self.subheaders = getattr(obj, 'subheaders', None)



    def __str__(self):

        if self.ndim == 1:

            arr = self[np.newaxis, :]

        elif self.ndim == 2:

            arr = self

        else:

            return super(TermValueArray, self).__str__()

        vsep_1col = '-' * 30 + '\n'

        vsep_2col = '-' * 65 + '\n'

        if arr.main_header:
        
            if arr.shape[0] == 1:
                
                s = vsep_1col

                s += '{0:^30}\n'.format(arr.main_header)

            else:

                s = vsep_2col

                s += '{0:^65}\n'.format(arr.main_header)

        else:

            s = ''

        m, n = arr.shape[0], arr.shape[1]

        for i in xrange(0, m - m % 2, 2):

            if arr.subheaders:

                s += vsep_2col

                s += ('{0:<20}{1:<15}{2:<20}{3}\n'
                      .format(arr.subheaders[i][0], 
                              arr.subheaders[i][1],
                              arr.subheaders[i+1][0], 
                              arr.subheaders[i+1][1]))
                                      
            s += vsep_2col

            for j in xrange(n):

                s += ('{0:<20}{1:<15.5f}{2:<20}{3:.5f}\n'
                      .format(arr[i][j][0], arr[i][j][1], 
                              arr[i+1][j][0], arr[i+1][j][1]))

        if m % 2:

            if arr.subheaders:

                s += vsep_1col

                s += ('{0:<20}{1}\n'
                      .format(arr.subheaders[m-1][0], 
                              arr.subheaders[m-1][1]))
                                      
            s += vsep_1col

            for j in xrange(n):

                s += ('{0:<20}{1:<15.5f}\n'
                      .format(arr[m-1][j][0], arr[m-1][j][1]))
            
        return s




def test_TermValueArray():

    terms = ['row', 'row', 'row', 'your', 'boat', 'gently', 'down', 'the', 
             'stream', 'merrily', 'merrily', 'merrily', 'merrily', 'life', 
             'is', 'but', 'a', 'dream']

    values = [np.random.random() for t in terms]

    d = [('term', np.array(terms).dtype), 
         ('value', np.array(values).dtype)]

    v = np.array(zip(terms, values), dtype=d)

    arr = np.vstack([v] * 5)

    arr = arr.view(TermValueArray)

    arr.main_header = 'Test 2-d Array'

    arr.subheaders = [('Repetition ' + str(i), 'Random') 
                      for i in xrange(arr.shape[0])]
    
    print arr

    print

    arr = v.view(TermValueArray)

    arr.main_header = 'Test 1-d Array'
    
    print arr
