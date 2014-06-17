import numpy as np

from vsm import (enum_sort as _enum_sort_, 
                 map_strarr as _map_strarr_)

from vsm.linalg import angle_sparse as _angle_sparse_

from vsm.viewer import (
    def_label_fn as _def_label_fn_,
    res_word_type as _res_word_type_)

from similarity import (
    dist_word_word as _dist_word_word_,
    dist_doc_doc as _dist_doc_doc_,
    dismat_word as _dismat_word_,
    dismat_doc as _dismat_doc_)

from labeleddata import LabeledColumn as _LabeledColumn_



class TfViewer(object):
    """
    A class for viewing Term-Frequency model.

    :param corpus: Source of observed data.
    :type corpus: Corpus

    :param model: A Term-Frequency model.
    :type model: TfSeq or TfMulti object.

    :attributes:
        * **corpus** (Corpus object) - `corpus`
        * **model** (Tf object) - `model`

    :methods:
        * :doc:`tf_sim_word_word`
            Returns words sorted by the cosine values between a word or list
            of words and every word.
        * :doc:`tf_sim_doc_doc`
            Computes and sorts the cosine similarity values between a
            document or list of documents.
        * :doc:`tf_simmat_words`
            Calculates the similarity matrix for a given list of words.
        * :doc:`tf_simmat_docs`
            Calculates the similarity matrix for a given list of documents.
        * :doc:`coll_freq`
        * :doc:`coll_freqs`

    :See Also: :class:`vsm.model.tf.TfSeq`, :class:`vsm.model.tf.TfMulti`
    """
    def __init__(self, corpus, model):
        """
        """
        self.corpus = corpus
        self.model = model


    def dist_word_word(self, word_or_words, weights=None, 
                       filter_nan=True, print_len=10, as_strings=True,
                       dist_fn=_angle_sparse_, order='i'):
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
        return _dist_word_word_(word_or_words, self.corpus, self.model.matrix.T,
                                weights=weights, filter_nan=filter_nan, 
                                print_len=print_len, as_strings=True,
                                dist_fn=dist_fn, order=order)


    def dist_doc_doc(self, doc_or_docs, weights=None, print_len=10, 
                     filter_nan=True, label_fn=_def_label_fn_, as_strings=True,
                     dist_fn=_angle_sparse_, order='i'):
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
        return _dist_doc_doc_(doc_or_docs, self.corpus, 
                              self.model.context_type, self.model.matrix, 
                              weights=weights, print_len=print_len,
                              filter_nan=filter_nan, label_fn=label_fn, 
                              as_strings=True, 
                              dist_fn=dist_fn, order=order)
    

    def dismat_word(self, word_list, dist_fn=_angle_sparse_):
        """
        Calculates a distance matrix for a given list of words.

        :param word_list: A list of words whose similarity matrix is to be
            computed.
        :type word_list: list

        :returns: .....
            contains n x n matrix containing floats where n is the number of words
            in `word_list`.

        :See Also: :meth:`vsm.viewer.similarity.dismat_words`
        """
        
        return _dismat_word_(word_list, self.corpus, 
                             self.model.matrix.T.tocsc(), dist_fn=dist_fn)


    def dismat_doc(self, docs, dist_fn=_angle_sparse_):
        """
        Calculates a distance matrix for a given list of documents.

        :param docs: A list of documents whose similarity matrix is to be computed.
            Default is all the documents in the model.
        :type docs: list, optional
        
        :returns: ....
            contains n x n matrix containing floats where n is the number of documents.

        :See Also: :meth:`vsm.viewer.similarity.dismat_docs`
        """
        return _dismat_doc_(docs, self.corpus, self.model.context_type, 
                            self.model.matrix.tocsc(), dist_fn=dist_fn)


    def coll_freq(self, word):
        """
        """
        i,w = _res_word_type_(self.corpus, word)
        row = self.model.matrix.tocsr()[i, :].toarray()
        return row.sum()

    
    def coll_freqs(self, print_len=20, as_strings=True):
        """

        :param print_len: Length of words to display. Default is `20`.
        :type print_len: integer, optional

        :param as_strings: If `True`, words are represented as strings
            rather than their integer representation.
        :type as_strings: boolean, optional

        :returns: :class:`LabeledColumn`.
            A table with word and its counts.
        """
        freqs = self.model.matrix.tocsr().sum(1) 
        w_arr = _enum_sort_(freqs.view(np.ndarray)[:, 0])
        
        # Label data
        if as_strings:
            w_arr = _map_strarr_(w_arr, self.corpus.words, k='i', new_k='word')
        w_arr = w_arr.view(_LabeledColumn_)
        w_arr.col_header = 'Collection Frequencies'
        w_arr.subcol_headers = ['Word', 'Counts']
        w_arr.col_len = print_len

        return w_arr
