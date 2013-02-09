import numpy as np

#TODO: Documentation update.



def split_corpus(arr, indices):
    """
    """
    if len(indices) == 0:
        return arr

    out = np.split(arr, indices)

    if (indices >= arr.size).any():
        out = out[:-1]

    for i in xrange(len(out)):
        if out[i].size == 0:
            out[i] = np.array([], dtype=arr.dtype)

    return out



class BaseCorpus(object):
    """
    **This documentation is out-of-date.**
    
    A BaseCorpus object stores a corpus along with its tokenizations
    (e.g., as sentences, paragraphs or documents).

    BaseCorpus aims to provide an efficient method for viewing the
    corpus in tokenized form (i.e., without copying). It currently
    achieves this by storing the corpus as a numpy `ndarray`. Viewing
    a tokenization is carried out using the `view` facilities of numpy
    ndarrays. See documentation on `numpy.ndarray` for further
    details.

    Parameters
    ----------
    corpus : array-like
        Array, typically of strings or integers, of atomic terms (or
        tokens) making up the corpus.
    tok_data : list with 1-D array-like elements, optional
        Each element in `tok_data` is an array containing the indices
        marking the token boundaries. An element in `tok_data` is
        intended for use as a value for the `indices_or_sections`
        parameter in `numpy.split`. Elements of `tok_data` may also be
        1-D arrays whose elements are pairs, where the first element
        is a token boundary and the second element is metadata
        associated with that token preceding that boundary. For
        example, (250, 'dogs') might indicate that the 'article' token
        ending at the 250th term of the corpus is named 'dogs'.
        Default is `None`.
    tok_names : array-like, optional
        Each element in `tok_names` is a name of a tokenization in
        `tok_data`.
    dtype : data-type, optional
        The data-type used to interpret the corpus. If omitted, the
        data-type is determined by `numpy.asarray`. Default is `None`.

    Attributes
    ----------
    corpus : 1-D array
        Stores the value of the `corpus` parameter after it has been
        cast to a array of data-type `dtype` (if provided).
    terms : 1-D array
        The indexed set of atomic terms appearing in `corpus`.
        Computed on initialization by `_extract_terms`.
    tok_names: 1-D array-like

    tok_data: list of 1-D array-like

    Methods
    -------
    view_tokens
        Takes a name of tokenization and returns a view of the corpus
        tokenized accordingly.
    view_metadata

    Examples
    --------

    >>> corpus = ['the', 'dog', 'chased', 'the', 'cat',
                  'the', 'cat', 'ran', 'away']
    >>> tok_names = ['sentences']
    >>> tok_data = [[(5, 'transitive'), (9, 'intransitive')]]

    >>> from vsm.corpus import BaseCorpus
    >>> c = BaseCorpus(corpus, tok_names=tok_names, tok_data=tok_data)
    >>> c.corpus
    array(['the', 'dog', 'chased', 'the', 'cat', 'the', 'cat',
           'ran', 'away'], dtype='|S6')
           
    >>> c.terms
    array(['ran', 'away', 'chased', 'dog', 'cat', 'the'],
          dtype='|S6')

    >>> c.view_tokens('sentences')
    [array(['the', 'dog', 'chased', 'the', 'cat'],
          dtype='|S6'),
     array(['the', 'cat', 'ran', 'away'],
          dtype='|S6')]

    >>> c.view_metadata('sentences')[0]
    'transitive'

    """
    def __init__(self,
                 corpus,
                 dtype=None,
                 tok_names=[],
                 tok_data=[]):

        self.corpus = np.asarray(corpus, dtype=dtype)
        self.dtype = self.corpus.dtype

        self._extract_terms()

        self.tok_data = []
        for t in tok_data:
            if self._validate_indices(t['idx']):
                self.tok_data.append(t)

        self._gen_tok_names(tok_names)


    def _gen_tok_names(self, tok_names):
        """
        Missing token names are filled in with 'tok_' + an index.
        """
        if self.tok_data:
            a = len(tok_names) if tok_names else 0
            for i in xrange(a, len(self.tok_data)):
                tok_names.append('tok_' + str(i))

        self.tok_names = tok_names


    def _validate_indices(self, indices):
        """
        Checks for invalid tokenizations. Specifically, checks to see
        that the list of indices are sorted and are in range. Ignores
        empty tokens.

        Parameters
        ----------
        indices : 1-D integer array-like

        Returns
        -------
        True if the indices are validated

        Raises
        ------
        Exception

        See Also
        --------
        BaseCorpus
        """
        for i, j in enumerate(indices):
            if i < len(indices) - 1 and j > indices[i + 1]:
                msg = 'malsorted tokenization:'\
                      ' tok ' + str(j) + ' and ' + str(indices[i + 1])

                raise Exception(msg)
                    
            if j > self.corpus.shape[0]:
                msg = 'invalid tokenization'\
                      ' : ' + str(j) + ' is out of range ('\
                      + str(self.corpus.shape[0]) + ')'
                
                raise Exception(msg)

        return True



    def view_metadata(self, name):
        """
        Displays the metadata corresponding to a tokenization of the
        corpus.

        Parameters
        ----------
        name : string-like
            The name of a tokenization.

        Returns
        -------
        The metadata for a tokenization.

        See Also
        --------
        BaseCorpus
        """
        i = self.tok_names.index(name)
        return self.tok_data[i]


    def meta_int(self, tok_name, query):
        """
        """
        tok = self.view_metadata(tok_name)

        ind_set = np.ones(tok.size, dtype=bool)
        for k,v in query.iteritems():
            ind_set = np.logical_and(ind_set, (tok[k] == v))

        n = np.count_nonzero(ind_set)
        if n == 0:
            raise Exception('No token fits that description.')
        elif n > 1:
            msg = ('Multiple tokens fit that description:\n'
                   + str(tok[ind_set]))
            raise Exception(msg)

        return ind_set.nonzero()[0][0]


    def get_metadatum(self, tok_name, query, field):
        """
        """
        i = self.meta_int(tok_name, query)
        return self.view_metadata(tok_name)[i][field]


    def view_tokens(self, name):
        """
        Displays a tokenization of the corpus.

        Parameters
        ----------
        name : string-like
           The name of a tokenization.

        Returns
        -------
        A tokenized view of `corpus`.

        See Also
        --------
        BaseCorpus
        numpy.split

        """
        i = self.tok_names.index(name)
        indices = self.tok_data[i]['idx']
        
        return split_corpus(self.corpus, indices)


    
    def _extract_terms(self):
        """
        Produces an indexed set of terms from a corpus.
        
        Parameters
        ----------
        corpus : array-like
        
        Returns
        -------
        An indexed set of the elements in `corpus` as a 1-D array.
        
        See Also
        --------
        BaseCorpus
        
        Notes
        -----
        """

        # Benchmarked by Peter Bengtsson
        # (http://www.peterbe.com/plog/uniqifiers-benchmark)
        
        term_set = set()
        term_list = [term for term in self.corpus
                     if term not in term_set and not term_set.add(term)]
        self.terms = np.array(term_list, dtype=self.corpus.dtype)



class Corpus(BaseCorpus):
    """
    **This documentation is out-of-date.**
    
    The goal of the Corpus class is to provide an efficient
    representation of a textual corpus.

    A Corpus object contains an integer representation of the text and
    maps to permit conversion between integer and string
    representations of a given term.

    As a BaseCorpus object, it includes a dictionary of tokenizations
    of the corpus and a method for viewing (without copying) these
    tokenizations. This dictionary also stores metadata (e.g.,
    document names) associated with the available tokenizations.

    Parameters
    ----------
    corpus : array-like
        A string array representing the corpus as a sequence of atomic
        terms.
    tok_data : list-like with 1-D integer array-like elements, optional
        Each element in `tok_data` is an array containing the indices
        marking the token boundaries. An element in `tok_data` is
        intended for use as a value for the `indices_or_sections`
        parameter in `numpy.split`. Elements of `tok_data` may also be
        1-D arrays whose elements are pairs, where the first element
        is a token boundary and the second element is metadata
        associated with that token preceding that boundary. For
        example, (250, 'dogs') might indicate that the 'article' token
        ending at the 250th term of the corpus is named 'dogs'.
        Default is `None`.
    tok_names : array-like, optional
        Each element in `tok_names` is a name of a tokenization in
        `tok_data`.

    Attributes
    ----------
    corpus : 1-D 32-bit integer array
        corpus is the integer representation of the input string
        array-like value value of the corpus parameter
    terms : 1-D string array
        The indexed set of strings occurring in corpus. It is a
        string-typed array.
    terms_int : 1-D 32-bit integer array
        A dictionary whose keys are `terms` and whose values are their
        corresponding integers (i.e., indices in `terms`).
    tok : dict with 1-D numpy arrays as values
        The tokenization dictionary. Stems of key names are given by
        `tok_names`. A key name whose value is the array of indices
        for a tokenization has the suffix '_indices'. A key name whose
        value is the metadata array for a tokenization has the suffix
        '_metadata'.
        
    Methods
    -------
    view_tokens
        Takes a name of tokenization and returns a view of the corpus
        tokenized accordingly. The optional parameter `strings` takes
        a boolean value: True to view string representations of terms;
        False to view integer representations of terms. Default is
        `False`.
    extract_terms
        Static method. Takes an array-like object and returns an
        indexed set of the elements in the object as a 1-D numpy
        array.
    gen_lexicon
        Returns a copy of itself but with `corpus`, `tokens`, and
        `tokens_meta` set to None. Occasionally, the only information
        needed from the Corpus object is the mapping between string
        and integer representations of terms; this provides a smaller
        version of the corpus object for such situations.
    save
        Takes a filename and saves the data contained in a Corpus
        object to a `npy` file using `numpy.savez`.
    load
        Static method. Takes a filename, loads the file data into a
        Corpus object and returns the object
    
    See Also
    --------
    BaseCorpus

    Examples
    --------

    >>> text = ['I', 'came', 'I', 'saw', 'I', 'conquered']
    >>> tok_names = ['sentences']
    >>> tok_data = [[(2, 'Veni'), (4, 'Vidi'), (6, 'Vici')]]

    >>> from vsm.corpus import Corpus
    >>> c = Corpus(text, tok_names=tok_names, tok_data=tok_data)
    >>> c.corpus
    array([0, 3, 0, 2, 0, 1], dtype=int32)
    
    >>> c.terms
    array(['I', 'conquered', 'saw', 'came'],
          dtype='|S9')

    >>> c.terms_int['saw']
    2

    >>> c.view_tokens('sentences')
    [array([0, 3], dtype=int32), array([0, 2], dtype=int32),
     array([0, 1], dtype=int32)]

    >>> c.view_tokens('sentences', strings=True)
    [array(['I', 'came'],
          dtype='|S4'), array(['I', 'saw'],
          dtype='|S3'), array(['I', 'conquered'],
          dtype='|S9')]

    >>> c.view_metadata('sentences')[1]
    'Vidi'
    
    """
    
    def __init__(self,
                 corpus,
                 tok_names=[],
                 tok_data=[]):

        super(Corpus, self).__init__(corpus,
                                     tok_names=tok_names,
                                     tok_data=tok_data,
                                     dtype=np.str_)

        self.__set_terms_int()

        # Integer encoding of a string-type corpus
        self.dtype = np.int32
        self.corpus = np.asarray([self.terms_int[term]
                                  for term in self.corpus],
                                 dtype=self.dtype)



    def __set_terms_int(self):
        """
        Mapping of terms to their integer representations.
        """
        self.terms_int = dict((t,i) for i,t in enumerate(self.terms))


    def view_tokens(self, name, as_strings=False):
        """
        Displays a tokenization of the corpus.

        Parameters
        ----------
        name : string-like
           The name of a tokenization.
        strings : Boolean, optional
            If True, string representations of terms are returned.
            Otherwise, integer representations are returned. Default
            is `False`.

        Returns
        -------
        A tokenized view of `corpus`.

        See Also
        --------
        Corpus
        BaseCorpus
        """
        token_list = super(Corpus, self).view_tokens(name)

        if as_strings:
            token_list_ = []
            for token in token_list:
                token = self.terms[token]
                token_list_.append(token)

            return token_list_

        return token_list


    @staticmethod
    def load(file):
        """
        Loads data into a Corpus object that has been stored using
        `save`.
        
        Parameters
        ----------
        file : str-like or file-like object
            Designates the file to read. If `file` is a string ending
            in `.gz`, the file is first gunzipped. See `numpy.load`
            for further details.

        Returns
        -------
        A Corpus object storing the data found in `file`.

        See Also
        --------
        Corpus
        Corpus.save
        numpy.load
        """
        print 'Loading corpus from', file
        arrays_in = np.load(file)

        c = Corpus([])
        c.corpus = arrays_in['corpus']
        c.terms = arrays_in['terms']
        c.tok_names = arrays_in['tok_names'].tolist()

        c.tok_data = list()
        for n in c.tok_names:
            t = arrays_in['tok_data_' + n]
            c.tok_data.append(t)

        c.__set_terms_int()

        return c


    def save(self, file):
        """
        Saves data from a Corpus object as an `npz` file.
        
        Parameters
        ----------
        file : str-like or file-like object
            Designates the file to which to save data. See
            `numpy.savez` for further details.
            
        Returns
        -------
        None

        See Also
        --------
        Corpus
        Corpus.load
        numpy.savez
        """
        print 'Saving corpus as', file
        arrays_out = dict()
        arrays_out['corpus'] = self.corpus
        arrays_out['terms'] = self.terms
        arrays_out['tok_names'] = np.asarray(self.tok_names)

        for i,t in enumerate(self.tok_data):
            key = 'tok_data_' + self.tok_names[i]
            arrays_out[key] = t

        np.savez(file, **arrays_out)


    def apply_stoplist(self, stoplist=[], freq=0):
        """
        Takes a Corpus object and returns a copy of it with terms in the
        stoplist removed and with terms of frequency <= `freq1` removed.
        """
        if freq:
            #TODO: Use the TF model instead

            print 'Computing collection frequencies'
            cfs = np.zeros_like(self.terms, dtype=self.corpus.dtype)
    
            for term in self.corpus:
                cfs[term] += 1

            print 'Selecting terms of frequency <=', freq
            freq_stop = np.arange(cfs.size)[(cfs <= freq)]
            stop = set(freq_stop)
        else:
            stop = set()

        for t in stoplist:
            if t in self.terms:
                stop.add(self.terms_int[t])

        if not stop:
            print 'Stop list is empty.'
            return self
    
        print 'Removing stop words'
        f = np.vectorize(lambda x: x not in stop)
        corpus = self.corpus[f(self.corpus)]

        print 'Rebuilding corpus'
        corpus = [self.terms[i] for i in corpus]
        tok_data = []
        for i in xrange(len(self.tok_data)):
            print 'Recomputing token breaks:', self.tok_names[i]
            tokens = self.view_tokens(self.tok_names[i])
            spans = [t[f(t)].size for t in tokens]
            tok = self.tok_data[i].copy()
            tok['idx'] = np.cumsum(spans)
            tok_data.append(tok)

        return Corpus(corpus, tok_data=tok_data, tok_names=self.tok_names)



#
# Testing
#


    
def test_file():

    from vsm.util.corpustools import random_corpus

    c = random_corpus(10000, 500, 1, 20, tok_name='foo', metadata=True)

    from tempfile import NamedTemporaryFile
    import os

    try:
        tmp = NamedTemporaryFile(delete=False, suffix='.npz')
        c.save(tmp.name)
        tmp.close()
        c_reloaded = c.load(tmp.name)

        assert (c.corpus == c_reloaded.corpus).all()
        assert (c.terms == c_reloaded.terms).all()
        assert c.terms_int == c_reloaded.terms_int
        assert c.tok_names == c_reloaded.tok_names
        for i in xrange(len(c.tok_data)):
            assert (c.tok_data[i] == c_reloaded.tok_data[i]).all()
    
    finally:
        os.remove(tmp.name)

    return c_reloaded
