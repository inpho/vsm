import numpy as np

from vsm import enum_sort
from vsm import corpus as cps
from vsm.corpus import util
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

    doc_labels = corpus.view_metadata(tok_name)['short_label'][doc_queries]
        
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

    return sim_vals



def similar_documents(corpus, matrix, tok_name, doc_query,
                      norms=None, filter_nan=True):

    if isinstance(doc_query, dict):
        
        doc_query = corpus.meta_int(tok_name, doc_query)
    
    sim_vals = similarity.similar_columns(doc_query, matrix, norms=norms,
                                          filter_nan=filter_nan)

    docs = corpus.view_metadata(tok_name)['short_label']

    sim_vals = np.array([(docs[i], v) for i,v in sim_vals],
                        dtype=[('doc', docs.dtype),
                               ('value', sim_vals.dtype['v'])])

    return sim_vals



def mean_similar_terms(corpus, matrix, query,
                       norms=None, filter_nan=True,
                       rem_masked=True):

    terms = util.word_tokenize(query)

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

    return sim_vals
