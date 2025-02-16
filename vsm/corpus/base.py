from __future__ import print_function
from builtins import str
from builtins import range
from builtins import object

from past.builtins import basestring

from itertools import tee
import os

import numpy as np
from sortedcontainers import SortedSet

from vsm.structarr import arr_add_field
from vsm.split import split_corpus


__all__ = [ 'BaseCorpus', 'Corpus', 'add_metadata',
            'align_corpora','binary_search' ]

from bisect import bisect_left
from datetime import datetime
from copy import deepcopy



def binary_search(a, x, lo=0, hi=None):   # can't use a to specify default for hi
    hi = hi if hi is not None else len(a) # hi defaults to len(a)   
    pos = bisect_left(a,x,lo,hi)          # find insertion position
    return (pos if pos != hi and a[pos] == x else -1) # don't walk off the end
"""
def binary_search_set(a,x):
    pos = a.bisect_left(x)
    return (pos if pos != len(a)  and a[pos] == x else -1) # don't walk off the end
"""

def load_npz(filename, obj):
    zipfile = np.load(filename)
    return zipfile.__getitem__(obj)

class BaseCorpus(object):
    """
    A BaseCorpus object stores a corpus along with its tokenizations
    (e.g., as sentences, paragraphs or documents).

    BaseCorpus aims to provide an efficient method for viewing the
    corpus in tokenized form (i.e., without copying). It currently
    achieves this by storing the corpus as a numpy `ndarray`. Viewing
    a tokenization is carried out using the `view` facilities of numpy
    ndarrays. See documentation on `numpy.ndarray` for further
    details.

    :param corpus: Array, typically of strings or integers, of atomic words
        (or tokens) making up the corpus.
    :type corpus: array-like

    :param dtype: The data-type used to interpret the corpus. If omitted, the
        data-type is determined by `numpy.asarray`. Default is `None`.
    :type dtype: data-type, optional

    :param context_data: Each element in `context_data` is an array 
        containing the indices marking the context boundaries. 
        An element in `context_data` is intended for use as a value for 
        the `indices_or_sections` parameter in `numpy.split`. 
        Elements of `context_data` may also be 1-D arrays whose elements
        are pairs, where the first element is a context boundary and 
        the second element is metadata associated with that context preceding
        that boundary. For example, (250, 'dogs') might indicate that 
        the 'article' context ending at the 250th word of the corpus is named 
        'dogs'. Default is `None`.
    :type context_data:  list with 1-D array-like elements, optional

    :param context_types: Each element in `context_types` is a type of a
        tokenization in `context_data`.
    :type context_types: array-like, optional

    :param remove_empty: If True, empty tokenizations are removed. Default is
        `True`.
    :type remove_empty: boolean, optional

    :param to_array: If True, converts all values to a numpy array. If False,
        data is left in input format. False is used when `BaseCorpus` is used
        in a constructor for an advanced tokenization like `Corpus`. Default is
        `True`.
    :type to_array: boolean, optional

    :attributes: 
        * **corpus**  (1-dimensional array)
            Stores the value of the `corpus` parameter after it has been cast
            to an array of data-type `dtype` (if provided).
        * **words**  (1-dimensional array)
            The indexed set of atomic words appearing in `corpus`.
        * **context_types**  (1-dimensional array-like)
        * **context_data**   (list of 1-D array-like)

    :methods: 
        * **meta_int**
            Takes a type of tokenization and a query and returns the index of
            the metadata found in the query.
        * **get_metadatum**
            Takes a type of tokenization and a query and returns the metadatum
            corresponding to the query and the field.
        * **view_contexts**
            Takes a type of tokenization and returns a view of the corpus
            tokenized accordingly.
        * **view_metadata**
            Takes a type of tokenization and returns a view of the metadata
            of the tokenization.
        * **tolist**
            Returns Corpus object as a list of lists.
        * **remove_empty**
            Removes empty documents in the corpus.

    :See Also: :class:`Corpus`

    **Examples**

    >>> corpus = ['the', 'dog', 'chased', 'the', 'cat',
                  'the', 'cat', 'ran', 'away']
    >>> context_types = ['sentences']
    >>> context_data = [np.array([(5, 'transitive'), (9, 'intransitive')],
                        dtype=[('idx', '<i8'), ('sent_label', '|S16')])]

    >>> from vsm.corpus import BaseCorpus
    >>> c = BaseCorpus(corpus, context_types=context_types, context_data=context_data)
    >>> c.corpus
    array(['the', 'dog', 'chased', 'the', 'cat', 'the', 'cat',
           'ran', 'away'], dtype='|S6')
           
    >>> c.words
    array(['ran', 'away', 'chased', 'dog', 'cat', 'the'],
          dtype='|S6')

    >>> c.meta_int('sentences',{'sent_label': 'intransitive'})
    1

    >>> b.get_metadatum('sentences', {'sent_label': 'intransitive'}, 'sent_label') 
    'intransitive'

    >>> c.view_contexts('sentences')
    [array(['the', 'dog', 'chased', 'the', 'cat'],
          dtype='|S6'),
     array(['the', 'cat', 'ran', 'away'],
          dtype='|S6')]

    >>> c.view_metadata('sentences')[0]['sent_label']
    'transitive'
    """
    __slots__ = ['words', 'context_data', 'corpus','dtype','stopped_words',
                 'context_types', 'original_length']
    def __init__(self,
                 corpus,
                 dtype=None,
                 context_types=[],
                 context_data=[],
                 remove_empty=False,
                 to_array=True,
                 words_corpus=None):

        if to_array:
            self.corpus = np.asarray(corpus, dtype=dtype)
            self.dtype = self.corpus.dtype
            self.words = np.unique(self.corpus)
        else:
            self.corpus = corpus
            self.dtype = dtype
            self.words = np.asarray(
                SortedSet(words_corpus if words_corpus else self.corpus), 
                dtype=np.object_)

        if hasattr(self.corpus, '__len__'):
            self._append_context_types(context_data, context_types)
            self.original_length = len(self.corpus)

        self.stopped_words = set()
        if remove_empty:
            self.remove_empty()
    
    def _append_context_types(self, context_data, context_types):
        self.context_data = []
        for t in context_data:
            if self._validate_indices(t['idx']):
                self.context_data.append(t)

        self._gen_context_types(context_types)


    def __len__(self):
        """
        Returns the number of tokens in the corpus.

        :See Also: `len(self.words)` for the number of unique tokens.
        """

        return len(self.corpus)

    def _gen_context_types(self, context_types):
        """
        Missing context types are filled in with 'ctx_' + an index.
        """
        if self.context_data:
            a = len(context_types) if context_types else 0
            for i in range(a, len(self.context_data)):
                context_types.append('ctx_' + str(i))

        self.context_types = context_types


    def _validate_indices(self, indices):
        """
        Checks for invalid tokenizations. Specifically, checks to see
        that the list of indices are sorted and are in range. Ignores
        empty tokens.

        :param indices: 
        :type indices : 1-D integer array-like

        :returns: `True` if the indices are validated

        :raises: Exception

        :See Also: :class:`BaseCorpus`
        """
        for i, j in enumerate(indices):
            if i < len(indices) - 1 and j > indices[i + 1]:
                msg = 'malsorted tokenization:'\
                      ' ctx ' + str(j) + ' and ' + str(indices[i + 1])

                raise Exception(msg)
                    
            if j > len(self.corpus):
                msg = 'invalid tokenization'\
                      ' : ' + str(j) + ' is out of range ('\
                      + str(len(self.corpus)) + ')'
                
                raise Exception(msg)

        return True

    
    def remove_empty(self):
        """
        Removes empty tokenizations, if `Corpus` object is not empty.
        """    
        if self:
            for j, t in enumerate(self.context_types):
                token_list = self.view_contexts(t)
                
                indices = np.array([ctx.size != 0 for ctx in token_list], dtype=bool)
                self.context_data[j] = self.context_data[j][indices]


    def view_metadata(self, ctx_type):
        """
        Displays the metadata corresponding to a tokenization of the
        corpus. This method can be used in :class:`Corpus` as well as
        :class:`BaseCorpus`

        :param ctx_type: The type of a tokenization.
        :type ctx_type: string-like

        :returns: The metadata for a tokenization.

        :See Also: :class:`BaseCorpus`, :class:`Corpus`
        """
        i = self.context_types.index(ctx_type)
        return self.context_data[i]


    def meta_int(self, ctx_type, query):
        """
        Returns the index of the metadata found in the query.

        :param ctx_type: The type of a tokenization.
        :type ctx_type: string-like
        
        :param query: Dictionary with a key, value being a field, label
            in metadata.
        :type query: dictionary-like

        :returns: The index of the metadata found in the query.
        
        :raises: KeyError

        :See Also: :class:`BaseCorpus`
        """

        tok = self.view_metadata(ctx_type)

        ind_set = np.ones(tok.size, dtype=bool)
        for k,v in query.items():
            try:
                ind_set = np.logical_and(ind_set, (tok[k] == v))
            except UnicodeDecodeError:
                v = v.decode('utf-8')
                ind_set = np.logical_and(ind_set, (tok[k] == v))

        n = np.count_nonzero(ind_set)
        if n == 0:
            raise KeyError('No token fits the description: ' +
                ', '.join(['{q}:{l}'.format(q=k, l=v) 
                                for k,v in query.items()]))
        elif n > 1:
            msg = ('Multiple tokens fit that description:\n'
                   + str(tok[ind_set]))
            raise KeyError(msg)

        return ind_set.nonzero()[0][0]


    def get_metadatum(self, ctx_type, query, field):
        """
        Returns the metadatum corresponding to the query and the field.

        :param ctx_type: The type of a tokenization.
        :type ctx_type: string-like
        
        :param query: Dictionary with a key, value being a field, label
            in metadata.
        :type query: dictionary-like
        
        :param field: Field of the metadata
        :type field: string

        :returns: The metadatum corresponding to the query and the field.

        :See Also: :class:`BaseCorpus`
        """
        i = self.meta_int(ctx_type, query)
        return self.view_metadata(ctx_type)[i][field]



    def view_contexts(self, ctx_type, as_slices=False, as_indices=False):
        """
        Displays a tokenization of the corpus.

        :param ctx_type: The type of a tokenization.
        :type ctx_type: string-like

        :param as_slices: If True, a list of slices corresponding to 
            'ctx_type' is returned. Otherwise, integer representations
            are returned. Default is `False`.
        :type as_slices: Boolean, optional

        :Returns: A tokenized view of `corpus`.

        :See Also: :class:`BaseCorpus`, :meth:`numpy.split`
        """
        indices = self.view_metadata(ctx_type)['idx']

        if as_indices:
            return indices
        
        if as_slices:
            if len(indices) == 0:
                return [slice(0, 0)]
                
            slices = []
            slices.append(slice(0, indices[0]))
            for i in range(len(indices) - 1):
                slices.append(slice(indices[i], indices[i+1]))
            return slices        
            
        return split_corpus(self.corpus, indices)


    def tolist(self, context_type):
        """
        Returns Corpus object as a list of lists.
        """
        return self.view_contexts(context_type)


    
class Corpus(BaseCorpus):
    """
    The goal of the Corpus class is to provide an efficient representation\
    of a textual corpus.

    A Corpus object contains an integer representation of the text and
    maps to permit conversion between integer and string
    representations of a given word.

    As a BaseCorpus object, it includes a dictionary of tokenizations
    of the corpus and a method for viewing (without copying) these
    tokenizations. This dictionary also stores metadata (e.g.,
    document names) associated with the available tokenizations.

    :param corpus: A string array representing the corpus as a sequence of
        atomic words.
    :type corpus: array-like

    :param context_data: Each element in `context_data` is an array containing 
        the indices marking the token boundaries. An element in `context_data` is
        intended for use as a value for the `indices_or_sections`
        parameter in `numpy.split`. Elements of `context_data` may also be
        1-D arrays whose elements are pairs, where the first element
        is a context boundary and the second element is metadata
        associated with that context preceding that boundary. For
        example, (250, 'dogs') might indicate that the 'article' context
        ending at the 250th word of the corpus is named 'dogs'.
        Default is `None`.
    :type context_data: list-like with 1-D integer array-like elements, optional
    
    :param context_types: Each element in `context_types` is a type of a context
        in `context_data`.
    :type context_types: array-like, optional

    :attributes: 
        * **corpus** (1-D 32-bit integer array)
            corpus is the integer representation of the input string array-like
            value of the corpus parameter
        * **words** (1-D string array)
            The indexed set of strings occurring in corpus. It is a string-typed array.
        * **words_in** (1-D 32-bit integer dictionary)
            A dictionary whose keys are `words` and whose values are their 
            corresponding integers (i.e., indices in `words`).
        
    :methods:
        * **view_metadata**
            Takes a type of tokenization and returns a view of the metadata
            of the tokenization.
        * **view_contexts**
            Takes a type of tokenization and returns a view of the corpus tokenized
            accordingly. The optional parameter `strings` takes a boolean value: 
            True to view string representations of words; False to view integer 
            representations of words. Default is `False`.
        * **save**
            Takes a filename and saves the data contained in a Corpus object to 
            a `npy` file using `numpy.savez`.
        * **load**
            Static method. Takes a filename, loads the file data into a Corpus
            object and returns the object.
        * **apply_stoplist**
            Takes a list of stopwords and returns a copy of the corpus with 
            the stopwords removed.
        * **tolist**
            Returns Corpus object as a list of lists of either integers or strings, 
            according to `as_strings`.
        
    :See Also: :class:`BaseCorpus`

    **Examples**

    >>> text = ['I', 'came', 'I', 'saw', 'I', 'conquered']
    >>> context_types = ['sentences']
    >>> context_data = [np.array([(2, 'Veni'), (4, 'Vidi'), (6, 'Vici')],
                            dtype=[('idx', '<i8'), ('sent_label', '|S6')])]

    >>> from vsm.corpus import Corpus
    >>> c = Corpus(text, context_types=context_types, context_data=context_data)
    >>> c.corpus
    array([0, 1, 0, 2, 0, 3], dtype=np.int32)
    
    >>> c.words
    array(['I', 'came', 'saw', 'conquered'],
          dtype='|S9')

    >>> c.words_int['saw']
    2

    >>> c.view_contexts('sentences')
    [array([0, 3], dtype=np.int32), array([0, 2], dtype=np.int32),
     array([0, 1], dtype=np.int32)]

    >>> c.view_contexts('sentences', as_strings=True)
    [array(['I', 'came'], 
          dtype='|S9'),
     array(['I', 'saw'], 
          dtype='|S9'),
     array(['I', 'conquered'], 
          dtype='|S9')]

    >>> c.view_metadata('sentences')[1]['sent_label']
    'Vidi'
    
    >>> c = c.apply_stoplist(['saw'])
    >>> c.words
    array(['I', 'came', 'conquered'], 
      dtype='|S9')

    """
    __slots__ = ['words', 'context_data', 'corpus','dtype','stopped_words',
                 'words_int', 'context_types', 'original_length']

    def __init__(self,
                 corpus,
                 context_types=[],
                 context_data=[],
                 remove_empty=True,
                 words_corpus=None):

        super(Corpus, self).__init__(corpus,
                                     context_types=context_types,
                                     context_data=context_data,
                                     dtype=np.object_,
                                     remove_empty=False,
                                     to_array=False,
                                     words_corpus=words_corpus)

        self._set_words_int()

        # Integer encoding of a string-type corpus
        if len(self.words) < 2 ** 16:
            self.dtype = np.uint16
        else:
            self.dtype = np.uint32

        self.corpus = np.fromiter(
            (self.words_int[word] for word in self.corpus),
            dtype=self.dtype)
            #count=len(self.corpus))
        
        self._append_context_types(context_data, context_types)

        self.stopped_words = set()
        self.original_length = len(self.corpus)

        if remove_empty:
            self.remove_empty()



    def _set_words_int(self):
        """
        Mapping of words to their integer representations.
        """
        self.words_int = dict((t,i) for i,t in enumerate(self.words))


    def view_contexts(self, ctx_type, as_strings=False, as_slices=False, as_indices=False):
        """
        Displays a tokenization of the corpus.

        :param ctx_type: The type of a tokenization.
        :type ctx_type: string-like

        :param as_strings: If True, string representations of words are returned.
            Otherwise, integer representations are returned. Default
            is `False`.
        :type as_strings: Boolean, optional

        :param as_slices: If True, a list of slices corresponding to 'ctx_type'
            is returned. Otherwise, integer representations are returned.
            Default is `False`.
        :type as_slices: Boolean, optional

        :returns: A tokenized view of `corpus`.

        :See Also: :class:`Corpus`, :class:`BaseCorpus`
        """     
        if as_strings:
            token_list = super(Corpus, self).view_contexts(ctx_type)
            token_list_ = []
            for token in token_list:
                token = self.words[token]
                token_list_.append(token)

            return token_list_

        return super(Corpus, self).view_contexts(ctx_type,
                                                 as_slices=as_slices,
                                                 as_indices=as_indices)
    
    
    def tolist(self, context_type, as_strings=False):
        """
        Returns Corpus object as a list of lists of either integers or
        strings, according to `as_strings`.

        :param context_type: The type of tokenization.
        :type context_type: string

        :param as_strings: If True, string representations of words are returned.
            Otherwise, integer representations are returned. Default
            is `False`.
        :type as_strings: Boolean, optional
        
        :returns: List of lists
        """
        ls = self.view_contexts(context_type, as_strings=as_strings)
        return [arr.tolist() for arr in ls]

    @staticmethod
    def load(file=None, corpus_dir=None,
             corpus_file='corpus.npy',
             words_file='words.npy',
             metadata_file='metadata.npy',
             load_corpus=True):
        """Loads data into a Corpus object. 
        
        :param file: The file to read. See `numpy.load` for further
            details. Assumes file has been constructed as by
            `Corpus.save`. This option is exclusive of `corpus_dir`.
        :type file: str-like or file object

        :param corpus_dir: A directory containing the files
        `corpus_file`, `words_file`, `metadata_file`, from which to
        instantiate a Corpus object. This option is ignored if `file`
        is not `None`.
        :type corpus_dir: string

        :param corpus_file: File under `corpus_dir` containing the
        corpus data, stored as a numpy array of integers in an `npy`
        file.
        :type corpus_file: string or file object
        
        :param words_file: File under `corpus_dir` containing the
        corpus vocabulary, stored as a numpy array of strings in an
        `npy` file.  
        :type words_file: string or file object
        
        :param metadata_file: File under `corpus_dir` containing the
        corpus metadata, stored as a numpy stuctured array in an `npy`
        file. Note that this structured array should contain a file
        `idx` which stores the integer indices marking the document
        boundaries.
        :type corpus_file: string or file object

        :param load_corpus: Determines whether to load the corpus. 
        If False, only the word list is loaded, greatly improving
        memory performance.
        :type load_corpus: boolean
        
        :returns: A Corpus object.

        :See Also: :class:`Corpus`, :meth:`Corpus.save`, :meth:`numpy.load`

        """


        try:
            __IPYTHON__
            notebook = True
            print("Running from notebook, using serial load function.")
        except NameError:
            notebook = True


        if not notebook and file is not None:
            return Corpus._parallel_load(file, load_corpus=load_corpus)

        elif notebook and file is not None:
            return Corpus._serial_load(file, load_corpus=load_corpus)

        elif not corpus_dir is None:
            return Corpus._multifile_load(corpus_dir=corpus_dir,
                corpus_file=corpus_file, words_file=words_file,
                metadata_file=metadata_file)


    @staticmethod
    def _parallel_load(file, load_corpus=True):
        import concurrent.futures
        import functools
        from pickle import PickleError, UnpicklingError

        c = Corpus([], remove_empty=False)
        # submit futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            if load_corpus:
                c.corpus = executor.submit(load_npz, file, 'corpus')
            c.words =  executor.submit(load_npz, file, 'words')

            c.context_types = executor.submit(load_npz, file, 'context_types')
            c.stopped_words = executor.submit(load_npz, file, 'stopped_words')
            c.original_length = executor.submit(load_npz, file, 'original_length')

            c.dtype = executor.submit(load_npz, file, 'dtype')
            concurrent.futures.wait([c.context_types])
            c.context_types = c.context_types.result().tolist()
            c.context_data = ['context_data_{}'.format(n) for n in c.context_types]
            c.context_data = [executor.submit(load_npz, file, name) for name in c.context_data]

        c.corpus = c.corpus.result() if load_corpus else None
        c.words = c.words.result()
        try:
            c.dtype = c.dtype.result()
        except KeyError:
            c.dtype = c.corpus.dtype
        try:
            c.original_length = int(c.original_length.result())
        except KeyError:
            c.original_length = None

        c.context_data = [future.result() for future in c.context_data]

        try:
            c.stopped_words = set(c.stopped_words.result().tolist())
        except:
            c.stopped_words = set()

        # reset words_int dictionary
        c._set_words_int()

        return c

    @staticmethod
    def _serial_load(file, load_corpus=True):
        arrays_in = np.load(file, allow_pickle=True, encoding='bytes')

        c = Corpus([], remove_empty=False)
        if load_corpus:
            c.corpus = arrays_in['corpus']
        else:
            c.corpus = None

        c.words = arrays_in['words']
        if isinstance(c.words[0], str):
            pass
        elif isinstance(c.words[0], basestring):
            c.words[:] = [w.decode('utf-8') for w in c.words]
        try:
            c.context_types = [ctx.decode('utf-8') 
                for ctx in arrays_in['context_types'].tolist()]
        except AttributeError:
            c.context_types = arrays_in['context_types'].tolist()

        try:
            c.stopped_words = set(arrays_in['stopped_words'].tolist())
        except:
            c.stopped_words = set()

        try:
            c.original_length = int(arrays_in['original_length'])
        except:
            c.original_length = None

        c.context_data = list()
        for n in c.context_types:
            try:
                n = n.decode('utf-8')
            except AttributeError:
                pass
            t = arrays_in['context_data_{}'.format(n)]
            
            if isinstance(t['{}_label'.format(n)][0], str):
                pass
            elif isinstance(t['{}_label'.format(n)][0], basestring):
                t['{}_label'.format(n)] = [lbl.decode('utf-8') for lbl in t['{}_label'.format(n)]]

            c.context_data.append(t)

        c._set_words_int()

        return c

    @staticmethod
    def _multifile_load(corpus_dir=None,
                        corpus_file='corpus.npy',
                        words_file='words.npy',
                        metadata_file='metadata.npy'):
        c = Corpus([], remove_empty=False)

        c.corpus = np.load(os.path.join(corpus_dir, corpus_file))
        c.words = np.load(os.path.join(corpus_dir, words_file))
        c._set_words_int()
        c.context_types = [ 'document' ]
        c.context_data = [ np.load(os.path.join(corpus_dir, metadata_file)) ]

        return c

    def save(self, file):
        """
        Saves data from a Corpus object as an `npz` file.
        
        :param file: Designates the file to which to save data. See
            `numpy.savez` for further details.
        :type file: str-like or file-like object
            
        :returns: None

        :See Also: :class:`Corpus`, :meth:`Corpus.load`, :meth:`np.savez`
        """
        print('Saving corpus as {}'.format(file))
        arrays_out = dict()
        arrays_out['corpus'] = self.corpus
        arrays_out['words'] = self.words
        arrays_out['context_types'] = np.asarray(self.context_types)
        arrays_out['stopped_words'] = np.asarray(self.stopped_words)
        arrays_out['dtype'] = str(self.dtype)
        arrays_out['original_length'] = int(self.original_length)

        for i,t in enumerate(self.context_data):
            key = 'context_data_' + self.context_types[i]
            arrays_out[key] = t

        np.savez(file, **arrays_out)


    def in_place_stoplist(self, stoplist=None, freq=0):
        """ 
        Changes a Corpus object with words in the stoplist removed and with 
        words of frequency <= `freq` removed.
        
        :param stoplist: The list of words to be removed.
        :type stoplist: list

        :param freq: A threshold where words of frequency <= 'freq' are
            removed. Default is 0.
        :type freq: integer, optional
            
        :returns: Copy of corpus with words in the stoplist and words
            of frequnecy <= 'freq' removed.

        :See Also: :class:`Corpus`
        """
        stop = np.zeros(self.words.shape, dtype=bool)

        if stoplist:
            for t in stoplist:
                if t in self.words_int:
                    stop[self.words_int[t]] = 1

        if freq:
            cfs = np.bincount(self.corpus)
            freq_stop = np.where(cfs <= freq)[0]
            stop[freq_stop] = 1


        if not stop.any():
            # print 'Stop list is empty.'
            return self
    

        # print 'Rebuilding context data', datetime.now()
        context_data = []

        BASE = len(self.context_data) - 1
        # gathering list of new indicies from narrowest tokenization
        def find_new_indexes(INTO, BASE=-1):
            locs = np.where(np.in1d(self.context_data[BASE]['idx'], self.context_data[INTO]['idx']))[0]

            # creating a list of lcoations that are non-identical
            new_locs = np.array([loc for i, loc in enumerate(locs)
                                     if i+1 == len(locs) or self.context_data[BASE]['idx'][locs[i]] != self.context_data[BASE]['idx'][locs[i+1]]])

            # creating a search for locations that ARE identical
            idxs = np.insert(self.context_data[INTO]['idx'], [0,-1], [-1,-1])
            same_spots = np.where(np.equal(idxs[:-1], idxs[1:]))[0]

            # readding the identical locations
            really_new_locs = np.insert(new_locs, same_spots, new_locs[same_spots-1])
            return really_new_locs

        # Calculate new base tokens
        tokens = self.view_contexts(self.context_types[BASE])
        spans = []
        for t in tokens:
            new_t = t[np.logical_not(stop)[t]] if t.size else t
            
            # TODO: append to new_corpus as well
            spans.append(new_t.size if new_t.size else 0)
        
        self.corpus = self.corpus[np.logical_not(stop)[self.corpus]]
        # Stopped all words from Corpus
        if not self.corpus.size:
            return Corpus([])

        new_base = self.context_data[BASE].copy()
        new_base['idx'] = np.cumsum(spans)

        context_data = []
        # calculate new tokenizations for every context_type
        for i in range(len(self.context_data)):
            if i == BASE:
                context_data.append(new_base)
            else:
                context = self.context_data[i].copy()
                context['idx'] = new_base['idx'][find_new_indexes(i, BASE)]
                context_data.append(context)

        del self.context_data
        self.context_data = context_data

        # print 'Rebuilding corpus and updating stop words', datetime.now()
        self.stopped_words.update(self.words[stop])

        #print 'adjusting words list', datetime.now()
        new_words = self.words[np.logical_not(stop)]
        old_to_new_array = np.zeros(self.words.shape, dtype=self.corpus.dtype)
        for i, word in enumerate(new_words):
            old_to_new_array[self.words_int[word]] = i

        # print 'rebuilding word dictionary', datetime.now()
        new_words_int = dict((word,i) for i, word in enumerate(new_words)) 

        #print "remapping corpus", datetime.now()
        self.corpus[:] = old_to_new_array[self.corpus]
        if (self.corpus.dtype != np.uint16 and len(new_words) < 2**16):
            self.corpus = self.corpus.astype(np.uint16)

        #print 'storing new word dicts', datetime.now()
        self.words = new_words
        self.words_int = new_words_int

        return self

    def apply_stoplist(self, stoplist=[], freq=0):
        """ 
        Takes a Corpus object and returns a copy of it with words in the
        stoplist removed and with words of frequency <= `freq` removed.
        
        :param stoplist: The list of words to be removed.
        :type stoplist: list

        :param freq: A threshold where words of frequency <= 'freq' are
            removed. Default is 0.
        :type freq: integer, optional
            
        :returns: Copy of corpus with words in the stoplist and words
            of frequnecy <= 'freq' removed.

        :See Also: :class:`Corpus`
        """
        new_c = deepcopy(self)
        return new_c.in_place_stoplist(stoplist=stoplist, freq=freq)

    def __deepcopy__(self, memo):
        _c = type(self)([], remove_empty=False)
        _c.corpus = np.copy(self.corpus)
        _c.words = np.copy(self.words)
        _c.context_types = self.context_types[:]
        _c.stopped_words = deepcopy(self.stopped_words, memo)
        _c.dtype = self.dtype
        _c.words_int = deepcopy(self.words_int, memo)
        _c.context_data = [np.copy(ctx) for ctx in self.context_data]

        return _c



def add_metadata(corpus, ctx_type, new_field, metadata):
    """
    Returns a corpus with metadata added.

    :param corpus: Corpus object to add new metadata to.
    :type corpus: :class:`Corpus`

    :param ctx_type: A type of tokenization.
    :type ctx_type: string

    :param new_field: Field name of the new metadata.
    :type new_field: string

    :param metadata: List of values to be added to `corpus`.
    :type metdata: list

    :returns: Corpus with new metadata added to the existing metdata.

    :See Also: :class:`Corpus`
    """
    i = corpus.context_types.index(ctx_type)
    md = corpus.context_data[i]
    corpus.context_data[i] = arr_add_field(md, new_field, metadata)

    return corpus



def align_corpora(old_corpus, new_corpus, remove_empty=True):
    """Takes two Corpus objects `old_corpus` and `new_corpus` and returns
    a copy of `new_corpus` with the following modifications: (1) the
    word to integer mapping agrees with that of `old_corpus` and (2)
    words in `new_corpus` which do not appear in `old_corpus` are
    removed from the corpus. Empty documents are removed.

    """
    new_words = [w for w in new_corpus.words if w not in old_corpus.words]
    out = new_corpus.apply_stoplist(new_words)
    if remove_empty:
        out.remove_empty()

    int_words = out.words
    words_int = old_corpus.words_int
    int_int = {}
    for i in range(len(int_words)):
        int_int[i] = words_int[int_words[i]]

    for i in range(len(out.corpus)):
        out.corpus[i] = int_int[out.corpus[i]]
    out.words = old_corpus.words.copy()
    out._set_words_int()

    return out

