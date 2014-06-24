import numpy as np

from vsm.spatial import angle
from vsm.structarr import *
from types import *
from labeleddata import *
from wrappers import *


__all__ = ['LsaViewer']


class LsaViewer(object):
    """
    A class for viewing LSA model.

    :param corpus: Source of observed data.
    :type corpus: Corpus

    :param model: An LSA mode.
    :type model: Lsa object.

    :attributes:
        * **corpus** (Corpus object) - `corpus`
        * **model** (Tf object) - `model`

    :methods:
        * :doc:`lsa_dist_word_word`
            Returns words sorted by the cosine values between a word or list
            of words and every word.
        * :doc:`lsa_dist_doc_doc`
            Computes and sorts the cosine similarity values between a
            document or list of documents and every document.
        * :doc:`lsa_simmat_words`
            Calculates the similarity matrix for a given list of words.
        * :doc:`lsa_simmat_docs`
            Calculates the similarity matrix for a given list of documents.

    :See Also: :mod:`vsm.model.lsa`
    """
    def __init__(self, corpus, model):
        """
        """
        self.corpus = corpus
        self.model = model


    def dist_word_word(self, word_or_words, weights=[], 
                       filter_nan=True, print_len=10, as_strings=True, 
                       dist_fn=angle, order='i'):
        """
        A wrapper of `dist_word_word` in similarity.py

        :param word_or_words: Query word(s) to which cosine values are calculated.
        :type word_or_words: string or list of strings
        
        :param weights: Specify weights for each query word in `word_or_words`. 
            Default uses equal weights (i.e. arithmetic mean)
        :type weights: list of floating point, optional
        
        :param filter_nan: If `True` not a number entries are filtered.
            Default is `True`.
        :type filter_nan: boolean, optional

        :param print_len: Number of words printed by pretty-printing function
            Default is 10.
        :type print_len: int, optional

        :param as_strings: If `True`, returns a list of words as strings rather
            than their integer representations. Default is `True`.
        :type as_strings: boolean, optional
        
        :returns: w_arr : :class:`LabeledColumn`.
            A 2-dim array containing words and their cosine values to 
            `word_or_words`. 
        
        :See Also: :meth:`vsm.viewer.similarity.dist_word_word`
        """
        return dist_word_word(word_or_words, self.corpus, 
                                self.model.word_matrix.T, weights=weights, 
                                filter_nan=filter_nan, 
                                print_len=print_len, as_strings=True, 
                                dist_fn=dist_fn, order=order)


    def dist_doc_doc(self, doc_or_docs, weights=[], print_len=10, 
                     filter_nan=True, label_fn=def_label_fn, as_strings=True,
                     dist_fn=angle, order='i'):
        """
        :param doc_or_docs: Query document(s) to which cosine values
            are calculated
        :type doc_or_docs: string/integer or list of strings/integers
        
        :param weights: Specify weights for each query doc in `doc_or_docs`. 
            Default uses equal weights (i.e. arithmetic mean)
        :type weights: list of floating point, optional
        
        :param print_len: Number of words printed by pretty-printing function.
            Default is 10.
        :type print_len: int, optional

        :param filter_nan: If `True` not a number entries are filtered.
            Default is `True`.
        :type filter_nan: boolean, optional
 
        :param label_fn: A function that defines how documents are represented.
            Default is def_label_fn which retrieves the labels from corpus metadata.
        :type label_fn: string, optional
        
        :param as_strings: If `True`, returns a list of words rather than
            their integer representations. Default is `True`.
        :type as_strings: boolean, optional

        :returns: w_arr : :class:`LabeledColumn`.
            A 2-dim array containing documents and their cosine values to 
            `doc_or_docs`. 
        
        :See Also: :meth:`vsm.viewer.similarity.dist_doc_doc`
        """
        return dist_doc_doc(doc_or_docs, self.corpus, self.model.context_type,
                              self.model.doc_matrix, weights=weights,
                              print_len=print_len, filter_nan=filter_nan, 
                              label_fn=label_fn, as_strings=True,
                              dist_fn=dist_fn, order=order)
    

    def dist_word_doc(self, word_or_words, weights=[], label_fn=def_label_fn, 
                      filter_nan=True, print_len=10, as_strings=True, 
                      dist_fn=angle, order='i'):
        """
        Computes distances between a word or a list of words to every
        document and sorts the results. The function constructs a
        pseudo-document vector from `word_or_words` and `weights`: the
        vector representation is non-zero only if the corresponding word
        appears in the list. If `weights` are not given, `1` is assigned
        to each word in `word_or_words`.
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


    def dismat_word(self, word_list, dist_fn=angle):
        """
        Calculates a distance matrix for a given list of words.

        :param word_list: A list of words whose similarity matrix is to be
            computed.
        :type word_list: list

        :returns: :class:`IndexedSymmArray`.
            contains n x n matrix containing floats where n is the number 
            of words in `word_list`.
        
        :See Also: :meth:`vsm.viewer.similarity.dismat_words`
        """
        return dismat_word(word_list, self.corpus, 
                             self.model.word_matrix.T, dist_fn=dist_fn)


    def dismat_doc(self, docs, dist_fn=angle):
        """
        Calculates a distance matrix for a given list of documents.

        :param docs: A list of documents whose similarity matrix is to be computed.
            Default is all the documents in the model.
        :type docs: list, optional
        
        :returns: :class:`IndexedSymmArray`.
            contains n x n matrix containing floats where n is the number of documents. 

        :See Also: :meth:`vsm.viewer.similarity.dismat_docs`
        """
        return dismat_doc(docs, self.corpus, self.model.context_type, 
                            self.model.doc_matrix, dist_fn=dist_fn)

