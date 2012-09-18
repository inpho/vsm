import numpy as np

from vsm import corpus as cps
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
                  norms=None, filter_nan=False):

    i = corpus.terms_int[term]
    
    cosines = similarity.similar_rows(i, matrix,
                                      norms=norms,
                                      filter_nan=filter_nan)
    
    return [(corpus.terms[t], v) for t,v in cosines]



def similar_documents(corpus, matrix, document,
                      norms=None, filter_nan=False):

    doc_names = corpus.view_metadata[tok_name]

    doc_names_int = zip(doc_names, xrange(len(doc_names)))
    
    i = doc_names_int[document]
    
    cosines = similarity.similar_columns(i, viewer.matrix,
                                         norms=norms,
                                         filter_nan=filter_nan)
    
    return [(doc_names[d], v) for d,v in cosines]
