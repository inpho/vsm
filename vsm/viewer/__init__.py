import numpy as np

from vsm import enum_sort
from vsm import corpus as cps
from vsm.util.corpustools import word_tokenize
from vsm import model

from similarity import (similar_rows, similar_columns, 
                        simmat_rows, simmat_columns)



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
    indices, terms = zip([res_term_type(corpus, term) 
                          for term in term_list])

    simmat = simmat_rows(matrix, indices)

    simmat.labels = terms
    
    return simmat



def simmat_documents(corpus, matrix, tok_name, doc_list):
    """
    """
    indices, labels = zip([res_doc_type(corpus, tok_name, doc) 
                           for doc in doc_list])

    simmat = simmat_columns(matrix, indices)

    simmat.labels = labels

    return simmat



def similar_terms(corpus, matrix, term,
                  norms=None, filter_nan=True,
                  rem_masked=True):
    """
    """
    i, term = res_term_type(corpus, term)
    
    t_arr = similar_rows(i, matrix,
                                       norms=norms,
                                       filter_nan=filter_nan)

    t_arr = np.array([(corpus.terms[i], v) for i,v in t_arr],
                        dtype=[('i', corpus.terms.dtype),
                               ('value', t_arr.dtype['value'])])

    if rem_masked:

        f = np.vectorize(lambda x: x is not np.ma.masked)

        t_arr = t_arr[f(t_arr['i'])]

    t_arr = t_arr.view(IndexedValueArray)

    t_arr.main_header = term

    t_arr.subheaders = [('Term', 'Cosine')]

    return t_arr



def similar_documents(corpus, matrix, tok_name, doc,
                      norms=None, filter_nan=True):
    """
    """
    d, label = res_doc_type(corpus, doc)

    d_arr = similar_columns(d, matrix, norms=norms,
                                          filter_nan=filter_nan)

    docs = corpus.view_metadata(tok_name)[doc_label_name(tok_name)]

    d_arr = np.array([(docs[i], v) for i,v in d_arr],
                     dtype=[('doc', docs.dtype), 
                            ('value', d_arr.dtype['value'])])

    d_arr = d_arr.view(IndexedValueArray)

    d_arr.main_header = label

    d_arr.subheaders = [('Document', 'Cosine')]

    return d_arr



def mean_similar_terms(corpus, matrix, query,
                       norms=None, filter_nan=True,
                       rem_masked=True):
    """
    """
    terms = word_tokenize(query)

    def sim_terms(term):

        i = corpus.terms_int[term]
    
        ra = similar_rows(i, matrix, norms=norms,
                                     sort=False, filter_nan=False)

        return ra['value']



    t_arr = [sim_terms(term) for term in terms]

    t_arr = reduce(np.add, t_arr)
    
    t_arr = t_arr / len(terms)

    t_arr = enum_sort(t_arr)
    
    if filter_nan:

        t_arr = t_arr[np.isfinite(t_arr['value'])]

    t_arr = np.array([(corpus.terms[i], v) for i,v in t_arr],
                        dtype=[('i', corpus.terms.dtype),
                               ('value', t_arr.dtype['value'])])

    if rem_masked:

        f = np.vectorize(lambda x: x is not np.ma.masked)

        t_arr = t_arr[f(t_arr['i'])]

    t_arr = t_arr.view(IndexedValueArray)

    t_arr.main_header = query

    t_arr.subheaders = [('Term', 'Cosine')]

    return t_arr



def format_entry(x):

    # np.void is the type of the tuples that appear in numpy
    # structured arrays
    if isinstance(x, np.void):

        return ', '.join([format_entry(i) for i in x.tolist()]) 
        
    if isinstance(x, basestring):

        return x

    if isinstance(x, int) or isinstance(x, long):

        return str(x)

    # Assume it's a float then
    return '{0:.5f}'.format(x)


        
class IndexedValueArray(np.ndarray):
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

            return super(IndexedValueArray, self).__str__()

        vsep_1col = '-' * 37 + '\n'

        vsep_2col = '-' * 75 + '\n'

        if arr.main_header:
        
            if arr.shape[0] == 1:
                
                s = vsep_1col

                s += '{0:^35}\n'.format(arr.main_header)

            else:

                s = vsep_2col

                s += '{0:^75}\n'.format(arr.main_header)

        else:

            s = ''

        m, n = arr.shape[0], arr.shape[1]

        for i in xrange(0, m - m % 2, 2):

            if arr.subheaders:

                s += vsep_2col

                s += ('{0:<25}{1:<15}{2:<25}{3}\n'
                      .format(arr.subheaders[i][0], 
                              arr.subheaders[i][1],
                              arr.subheaders[i+1][0], 
                              arr.subheaders[i+1][1]))
                                      
            s += vsep_2col

            for j in xrange(n):

                a0 = format_entry(arr[i][j][0])

                a1 = format_entry(arr[i][j][1])

                b0 = format_entry(arr[i+1][j][0])

                b1 = format_entry(arr[i+1][j][1])

                s += '{0:<25}{1:<15}{2:<25}{3}\n'.format(a0, a1, b0, b1)

        if m % 2:

            if arr.subheaders:

                s += vsep_1col

                s += ('{0:<25}{1}\n'
                      .format(arr.subheaders[m-1][0], 
                              arr.subheaders[m-1][1]))
                                      
            s += vsep_1col

            for j in xrange(n):

                a0 = format_entry(arr[m-1][j][0])

                a1 = format_entry(arr[m-1][j][1])

                s += '{0:<25}{1}\n'.format(a0, a1)
            
        return s



def doc_label_name(tok_name):

    return tok_name + '_label'



def res_doc_type(corpus, tok_name, label_name, doc):
    """
    If `doc` is a string or a dict, performs a look up for its
    associated integer. If `doc` is a dict, looks for its label.
    Finally, if `doc` is an integer, stringifies `doc` for use as
    a label. 
    
    Returns an integer, string pair: (<document index>, <document
    label>).
    """
    if isinstance(doc, basestring):
        
        query = {label_name: doc}
        
        d = corpus.meta_int(tok_name, query)
        
    elif isinstance(doc, dict):
        
        d = corpus.meta_int(tok_name, doc)
        
        #TODO: Define an exception for failed queries in
        #vsm.corpus. Use it here.
        
        doc = corpus.view_metadata(tok_name)[label_name][d]

    else:

        d, doc = doc, str(doc)

    return d, doc
                    
            

def res_term_type(corpus, term):
    """
    If `term` is a string, performs a look up for its associated
    integer. Otherwise, stringifies `term`. 

    Returns an integer, string pair: (<term index>, <term label>).
    """
    if isinstance(term, basestring):
            
        return corpus.terms_int[term], term

    return term, str(term)
        


############################################################
#                        Testing
############################################################

def test_IndexedValueArray():

    terms = ['row', 'row', 'row', 'your', 'boat', 'gently', 'down', 'the', 
             'stream', 'merrily', 'merrily', 'merrily', 'merrily', 'life', 
             'is', 'but', 'a', 'dream']

    values = [np.random.random() for t in terms]

    d = [('i', np.array(terms).dtype), 
         ('value', np.array(values).dtype)]

    v = np.array(zip(terms, values), dtype=d)

    arr = np.vstack([v] * 5)

    arr = arr.view(IndexedValueArray)

    arr.main_header = 'Test 2-d Array'

    arr.subheaders = [('Repetition ' + str(i), 'Random') 
                      for i in xrange(arr.shape[0])]
    
    print arr

    print

    arr = v.view(IndexedValueArray)

    arr.main_header = 'Test 1-d Array'
    
    print arr
