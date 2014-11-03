import numpy as np

from vsm.spatial import angle
from vsm.structarr import *
from vsm.exceptions import *
from types import *
from labeleddata import *
from wrappers import *


__all__ = ['LsaViewer']


class LsaViewer(object):
    """
    A class for viewing LSA model.
    """
    
    def __init__(self, corpus, model):
        """
        Initialize LsaViewer.

        :param corpus: Source of observed data.
        :type corpus: :class:`Corpus`

        :param model: An LSA model.
        :type model: Lsa
        """
        self.corpus = corpus
        self.model = model

    @deprecated_meth("dist_word_word")
    def sim_word_word(self, word_or_words, weights=[], 
                       filter_nan=True, print_len=10, as_strings=True, 
                       dist_fn=angle, order='i'):

        pass


    def dist_word_word(self, word_or_words, weights=[], 
                       filter_nan=True, print_len=10, as_strings=True, 
                       dist_fn=angle, order='i'):
        """
        Computes and sorts the distances between a word or list
        of words and every word.

        :param word_or_words: Query word(s) to which distances are calculated.
        :type word_or_words: string or list of strings
        
        :param weights: Specify weights for each query word in `word_or_words`. 
            Default uses equal weights (i.e. arithmetic mean)
        :type weights: list of floating point, optional
        
        :param filter_nan: If `True` not a number entries are filtered.
            Default is `True`.
        :type filter_nan: boolean, optional

        :param print_len: Number of words to be displayed. Default is 10.
        :type print_len: int, optional

        :param as_strings: If `True`, returns a list of words as strings rather
            than their integer representations. Default is `True`.
        :type as_strings: boolean, optional
        
        :param dist_fn: A distance function from functions in vsm.spatial. 
            Default is :meth:`angle`.
        :type dist_fn: string, optional
        
        :param order: Order of sorting. 'i' for increasing and 'd' for
            decreasing order. Default is 'i'.
        :type order: string, optional

        :returns: an instance of :class:`LabeledColumn`.
            A 2-dim array containing words and their distances to 
            `word_or_words`. 
        
        :See Also: :meth:`vsm.viewer.wrappers.dist_word_word`
        """
        return dist_word_word(word_or_words, self.corpus, 
                                self.model.word_matrix.T, weights=weights, 
                                filter_nan=filter_nan, 
                                print_len=print_len, as_strings=True, 
                                dist_fn=dist_fn, order=order)

    @deprecated_meth("dist_doc_doc")
    def sim_doc_doc(self, doc_or_docs, weights=[], print_len=10, 
                     filter_nan=True, label_fn=def_label_fn, as_strings=True,
                     dist_fn=angle, order='i'):
        pass
        
    def dist_doc_doc(self, doc_or_docs, weights=[], print_len=10, 
                     filter_nan=True, label_fn=def_label_fn, as_strings=True,
                     dist_fn=angle, order='i'):
        """
        Computes and sorts the distances between a
        document or list of documents and every document.

        :param doc_or_docs: Query document(s) to which distances
            are calculated.
        :type doc_or_docs: string/integer or list of strings/integers
        
        :param weights: Specify weights for each query doc in `doc_or_docs`. 
            Default uses equal weights (i.e. arithmetic mean)
        :type weights: list of floating point, optional
        
        :param print_len: Number of words to be displayed. Default is 10.
        :type print_len: int, optional

        :param filter_nan: If `True` not a number entries are filtered.
            Default is `True`.
        :type filter_nan: boolean, optional
 
        :param label_fn: A function that defines how documents are represented.
            Default is def_label_fn which retrieves the labels from corpus
            metadata.
        :type label_fn: string, optional
        
        :param as_strings: If `True`, returns a list of words rather than
            their integer representations. Default is `True`.
        :type as_strings: boolean, optional
        
        :param dist_fn: A distance function from functions in vsm.spatial. 
            Default is :meth:`angle`.
        :type dist_fn: string, optional
        
        :param order: Order of sorting. 'i' for increasing and 'd' for
            decreasing order. Default is 'i'.
        :type order: string, optional

        :returns: an instance of :class:`LabeledColumn`.
            A 2-dim array containing documents and their distances to 
            `doc_or_docs`. 
        
        :See Also: :meth:`vsm.viewer.wrappers.dist_doc_doc`
        """
        return dist_doc_doc(doc_or_docs, self.corpus, self.model.context_type,
                              self.model.doc_matrix, weights=weights,
                              print_len=print_len, filter_nan=filter_nan, 
                              label_fn=label_fn, as_strings=True,
                              dist_fn=dist_fn, order=order)
    
    @deprecated_meth("dist_word_doc")
    def sim_word_doc(self, word_or_words, weights=[], label_fn=def_label_fn, 
                      filter_nan=True, print_len=10, as_strings=True, 
                      dist_fn=angle, order='i'):
        pass
    
    def dist_word_doc(self, word_or_words, weights=[], label_fn=def_label_fn, 
                      filter_nan=True, print_len=10, as_strings=True, 
                      dist_fn=angle, order='i'):
        """
        Computes and sorts distances between a word or a list of words to
        every document.
        
        :param word_or_words: Query word(s) to which a pseudo-document is
            created for computation of distances.
        :type word_or_words: string/integer or list of strings/integers
        
        :param weights: Specify weights for each query doc in `word_or_words`. 
            Default uses equal weights (i.e. arithmetic mean)
        :type weights: list of floating point, optional
        
        :param print_len: Number of documents to be displayed. Default is 10.
        :type print_len: int, optional

        :param filter_nan: If `True` not a number entries are filtered.
            Default is `True`.
        :type filter_nan: boolean, optional
 
        :param label_fn: A function that defines how documents are represented.
            Default is :meth:`def_label_fn` which retrieves the labels 
            from corpus metadata.
        :type label_fn: string, optional
        
        :param as_strings: If `True`, returns a list of documents as strings
            rather than indices. Default is `True`.
        :type as_strings: boolean, optional

        :param dist_fn: A distance function from functions in vsm.spatial.
            Default is :meth:`angle`.
        :type dist_fn: string, optional
         
        :param order: Order of sorting 'i' for increasing and 'd' for
            decreasing order. Default is 'i'.
        :type order: string, optional
       
        :returns: an instance of :class:`LabeledColumn`.
            A 2-dim array containing documents and their distances to 
            `word_or_words`. 

        :See Also: :meth:`vsm.viewer.wrappers.dist_word_doc`
        """
        # Resolve `word_or_words`
        if isstr(word_or_words):
            word_or_words = [word_or_words]
        words, labels = zip(*[res_word_type(self.corpus, w) for w in word_or_words])
        words, labels = list(words), list(labels)

        # Generate pseudo-document
        doc = np.zeros((self.model.word_matrix.shape[0],1), dtype=np.float)
        if len(weights) == 0:
            doc[words,:] = np.ones(len(words))
        else:
            doc[words,:] = weights

        doc = np.dot(np.dot(np.diag(1 /self.model.eigenvalues), 
                            self.model.word_matrix.T), doc)

        # Compute distances
        d_arr = dist_fn(doc.T, self.model.doc_matrix)

        # Label data
        if as_strings:
            md = self.corpus.view_metadata(self.model.context_type)
            docs = label_fn(md)
            d_arr = enum_sort(d_arr, indices=docs, field_name='doc')
        else:
            d_arr = enum_sort(d_arr, filter_nan=filter_nan)

        if order=='d':
            pass
        elif order=='i':
            d_arr = d_arr[::-1]
        else:
            raise Exception('Invalid order parameter.')

        d_arr = d_arr.view(LabeledColumn)
        # TODO: Finish this header
        d_arr.col_header = 'Words: '
        d_arr.subcol_headers = ['Document', 'Distance']
        d_arr.col_len = print_len
        
        return d_arr

    @deprecated_meth("dismat_word")
    def simmat_words(self, word_list, dist_fn=angle):
        pass
    
    def dismat_word(self, word_list, dist_fn=angle):
        """
        Calculates a distance matrix for a given list of words.

        :param word_list: A list of words whose similarity matrix is to be
            computed.
        :type word_list: list

        :param dist_fn: A distance function from functions in vsm.spatial. 
            Default is :meth:`angle`.
        :type dist_fn: string, optional

        :returns: an instance of :class:`IndexedSymmArray`.
            n x n matrix containing floats where n is the number 
            of words in `word_list`.
        
        :See Also: :meth:`vsm.viewer.wrappers.dismat_word`
        """
        return dismat_word(word_list, self.corpus, 
                             self.model.word_matrix.T, dist_fn=dist_fn)

    @deprecated_meth("dismat_doc")
    def simmat_docs(self, doc_list, dist_fn=angle):
        pass
    
    def dismat_doc(self, doc_list, dist_fn=angle):
        """
        Calculates a distance matrix for a given list of documents.

        :param doc_list: A list of documents whose distance matrix is 
            to be computed.
        :type docs: list, optional
        
        :param dist_fn: A distance function from functions in vsm.spatial. 
            Default is :meth:`angle`.
        :type dist_fn: string, optional

        :returns: an instance of :class:`IndexedSymmArray`.
            n x n matrix containing floats where n is the number 
            of documents. 

        :See Also: :meth:`vsm.viewer.wrappers.dismat_doc`
        """
        return dismat_doc(doc_list, self.corpus, self.model.context_type, 
                            self.model.doc_matrix, dist_fn=dist_fn)

