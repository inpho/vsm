import numpy as np

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




def simmat_documents(corpus, matrix, tok_name, document_list):

    doc_names = corpus.view_metadata[tok_name]

    doc_names_int = zip(doc_names, xrange(len(doc_names)))

    labels = [doc_names_int[doc] for doc in document_list]
    
    simmat = similarity.simmat_columns(labels, matrix)

    simmat.labels = document_list

    return simmat



def similar_terms(corpus, matrix, term,
                  norms=None, filter_nan=True,
                  rem_masked=True):

    i = corpus.terms_int[term]
    
    sim_vals = similarity.similar_rows(i, matrix,
                                       norms=norms,
                                       filter_nan=filter_nan)
    
    out = []

    for t,v in sim_vals:

        term = corpus.terms[t]

        if not (rem_masked and term is np.ma.masked):

            out.append((term, v))

    return out



def similar_documents(corpus, matrix, document,
                      norms=None, filter_nan=True):

    doc_names = corpus.view_metadata[tok_name]

    doc_names_int = zip(doc_names, xrange(len(doc_names)))
    
    i = doc_names_int[document]
    
    cosines = similarity.similar_columns(i, viewer.matrix,
                                         norms=norms,
                                         filter_nan=filter_nan)
    
    return [(doc_names[d], v) for d,v in cosines]





def mean_similar_terms(corpus, matrix, query,
                       norms=None, filter_nan=True,
                       rem_masked=True):

    terms = util.word_tokenize(query)

    def sim_terms(term):

        i = corpus.terms_int[term]
    
        ra = similarity.similar_rows(i, matrix, norms=norms,
                                     sort=False, filter_nan=False)

        return ra['value']



    sim_vals = [sim_terms(term) for term in terms]

    sim_vals = reduce(np.add, sim_vals)
    
    sim_vals = sim_vals / len(terms)

    sim_vals = list(enumerate(sim_vals.tolist()))
    
    dtype = [('index', np.int), ('value', np.float)]

    sim_vals = np.array(sim_vals, dtype=dtype)

    sim_vals = similarity.sort_sim(sim_vals)

    if filter_nan:

        sim_vals = similarity._filter_nan(sim_vals)

    out = []

    for t,v in sim_vals:

        term = corpus.terms[t]

        if not (rem_masked and term is np.ma.masked):

            out.append((term, v))

    return out
