import numpy as np

from vsm.viewer import doc_label_name




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
        Array, typically of strings or integers, of atomic words (or
        tokens) making up the corpus.
    context_data : list with 1-D array-like elements, optional
        Each element in `context_data` is an array containing the indices
        marking the context boundaries. An element in `context_data` is
        intended for use as a value for the `indices_or_sections`
        parameter in `numpy.split`. Elements of `context_data` may also be
        1-D arrays whose elements are pairs, where the first element
        is a context boundary and the second element is metadata
        associated with that context preceding that boundary. For
        example, (250, 'dogs') might indicate that the 'article' context
        ending at the 250th word of the corpus is named 'dogs'.
        Default is `None`.
    context_types : array-like, optional
        Each element in `context_types` is a type of a tokenization in
        `context_data`.
    dtype : data-type, optional
        The data-type used to interpret the corpus. If omitted, the
        data-type is determined by `numpy.asarray`. Default is `None`.

    Attributes
    ----------
    corpus : 1-D array
        Stores the value of the `corpus` parameter after it has been
        cast to an array of data-type `dtype` (if provided).
    words : 1-D array
        The indexed set of atomic words appearing in `corpus`.
        Computed on initialization by `_extract_words`.
    context_types: 1-D array-like

    context_data: list of 1-D array-like

    Methods
    -------
    meta_int
	Takes a type of tokenization and a query and 
	returns the index of the metadata found in the query.
    get_metadatum
	Takes a type of tokenization and a query and returns 
	the metadatum corresponding to the query and the field.
    view_contexts
        Takes a type of tokenization and returns a view of the corpus
        tokenized accordingly.
    view_metadata
	Takes a type of tokenization and returns a view of the metadata
	of the tokenization.

    Examples
    --------

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

    >>> c.meta_int('sentences',{'sent_label': 'intransitive'}, 'sent_label')
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
    def __init__(self,
                 corpus,
                 dtype=None,
                 context_types=[],
                 context_data=[],
		 remove_empty=True):

        self.corpus = np.asarray(corpus, dtype=dtype)
        self.dtype = self.corpus.dtype

        self._extract_words()

        self.context_data = []
        for t in context_data:
            if self._validate_indices(t['idx']):
                self.context_data.append(t)

        self._gen_context_types(context_types)

	if remove_empty:
	    self.remove_empty()


    def _gen_context_types(self, context_types):
        """
        Missing context types are filled in with 'ctx_' + an index.
        """
        if self.context_data:
            a = len(context_types) if context_types else 0
            for i in xrange(a, len(self.context_data)):
                context_types.append('ctx_' + str(i))

        self.context_types = context_types


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
                      ' ctx ' + str(j) + ' and ' + str(indices[i + 1])

                raise Exception(msg)
                    
            if j > self.corpus.shape[0]:
                msg = 'invalid tokenization'\
                      ' : ' + str(j) + ' is out of range ('\
                      + str(self.corpus.shape[0]) + ')'
                
                raise Exception(msg)

        return True


    def remove_empty(self):
	"""
	Removes empty tokenizations.
	"""	
	for j, t in enumerate(self.context_types):
	    token_list = super(Corpus, self).view_contexts(t)

 	    indices = []
	    for i, ctx in enumerate(token_list):
	    	if len(ctx) < 1:
		    indices.append(i)

	    self.context_data[j] = np.delete(self.context_data[j], indices)	


    def view_metadata(self, ctx_type):
        """
        Displays the metadata corresponding to a tokenization of the
        corpus.

        Parameters
        ----------
        ctx_type : string-like
            The type of a tokenization.

        Returns
        -------
        The metadata for a tokenization.

        See Also
        --------
        BaseCorpus
        """
        i = self.context_types.index(ctx_type)
        return self.context_data[i]


    def meta_int(self, ctx_type, query):
        """
	Returns the index of the metadata found in the query.

	Parameters
        ----------
        ctx_type : string-like
            The type of a tokenization.
	query: dictionary-like
	    Dictionary with a key, value being a field, label in metadata.

        Returns
        -------
        The index of the metadata found in the query.

        See Also
        --------
        BaseCorpus
        """

	tok = self.view_metadata(ctx_type)

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


    def get_metadatum(self, ctx_type, query, field):
        """
	Returns the metadatum corresponding to the query and the field.

	Parameters
        ----------
        ctx_type : string-like
            The type of a tokenization.
	query : dictionary-like
	    Dictionary with a key, value being a field, label in metadata.
	field : string
	    Field of the metadata

        Returns
        -------
        The metadatum corresponding to the query and the field.

        See Also
        --------
        BaseCorpus
        """
        i = self.meta_int(ctx_type, query)
        return self.view_metadata(ctx_type)[i][field]


    def view_contexts(self, ctx_type, as_slices=False):
        """
        Displays a tokenization of the corpus.

        Parameters
        ----------
        ctx_type : string-like
           The type of a tokenization.
	as_slices : Boolean, optional
            If True, a list of slices corresponding to 'ctx_type' is 
	    returned. Otherwise, integer representations are returned.
	    Default is `False`.

        Returns
        -------
        A tokenized view of `corpus`.

        See Also
        --------
        BaseCorpus
        numpy.split

        """
        i = self.context_types.index(ctx_type)
        indices = self.context_data[i]['idx']

	if as_slices:
	    meta_list = self.view_metadata(ctx_type)
	    indices = meta_list['idx'] 

	    slices = []
	    slices.append(slice(0, indices[0]))
	    for i in xrange(len(indices) - 1):
		slices.append(slice(indices[i], indices[i+1]))
	    return slices	    
       
        return split_corpus(self.corpus, indices)


    
    def _extract_words(self):
        """
        Produces an indexed set of words from a corpus.
        
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
        
        word_set = set()
        word_list = [word for word in self.corpus
                     if word not in word_set and not word_set.add(word)]
        self.words = np.array(word_list, dtype=self.corpus.dtype)



class Corpus(BaseCorpus):
    """
    The goal of the Corpus class is to provide an efficient
    representation of a textual corpus.

    A Corpus object contains an integer representation of the text and
    maps to permit conversion between integer and string
    epresentations of a given word.

    As a BaseCorpus object, it includes a dictionary of tokenizations
    of the corpus and a method for viewing (without copying) these
    tokenizations. This dictionary also stores metadata (e.g.,
    document names) associated with the available tokenizations.

    Parameters
    ----------
    corpus : array-like
        A string array representing the corpus as a sequence of atomic
        words.
    context_data : list-like with 1-D integer array-like elements, optional
        Each element in `context_data` is an array containing the indices
        marking the token boundaries. An element in `context_data` is
        intended for use as a value for the `indices_or_sections`
        parameter in `numpy.split`. Elements of `context_data` may also be
        1-D arrays whose elements are pairs, where the first element
        is a context boundary and the second element is metadata
        associated with that context preceding that boundary. For
        example, (250, 'dogs') might indicate that the 'article' context
        ending at the 250th word of the corpus is named 'dogs'.
        Default is `None`.
    context_types : array-like, optional
        Each element in `context_types` is a type of a context in
        `context_data`.

    Attributes
    ----------
    corpus : 1-D 32-bit integer array
        corpus is the integer representation of the input string
        array-like value value of the corpus parameter
    words : 1-D string array
        The indexed set of strings occurring in corpus. It is a
        string-typed array.
    words_int : 1-D 32-bit integer dictionary
        A dictionary whose keys are `words` and whose values are their
        corresponding integers (i.e., indices in `words`).
        
    Methods
    -------
    view_contexts
        Takes a type of tokenization and returns a view of the corpus
        tokenized accordingly. The optional parameter `strings` takes
        a boolean value: True to view string representations of words;
        False to view integer representations of words. Default is
        `False`.
    save
        Takes a filename and saves the data contained in a Corpus
        object to a `npy` file using `numpy.savez`.
    load
        Static method. Takes a filename, loads the file data into a
        Corpus object and returns the object.
    apply_stoplist
	Takes a list of stopwords and returns a copy of the corpus
	with the stopwords removed.
    
    See Also
    --------
    BaseCorpus

    Examples
    --------

    >>> text = ['I', 'came', 'I', 'saw', 'I', 'conquered']
    >>> context_types = ['sentences']
    >>> context_data = [np.array([(2, 'Veni'), (4, 'Vidi'), (6, 'Vici')],
                            dtype=[('idx', '<i8'), ('sent_label', '|S6')])]

    >>> from vsm.corpus import Corpus
    >>> c = Corpus(text, context_types=context_types, context_data=context_data)
    >>> c.corpus
    array([0, 1, 0, 2, 0, 3], dtype=int32)
    
    >>> c.words
    array(['I', 'came', 'saw', 'conquered'],
          dtype='|S9')

    >>> c.words_int['saw']
    2

    >>> c.view_contexts('sentences')
    [array([0, 3], dtype=int32), array([0, 2], dtype=int32),
     array([0, 1], dtype=int32)]

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
    
    def __init__(self,
                 corpus,
                 context_types=[],
                 context_data=[]):

        super(Corpus, self).__init__(corpus,
                                     context_types=context_types,
                                     context_data=context_data,
                                     dtype=np.str_)

        self.__set_words_int()

        # Integer encoding of a string-type corpus
        self.dtype = np.int32
        self.corpus = np.asarray([self.words_int[word]
                                  for word in self.corpus],
                                 dtype=self.dtype)



    def __set_words_int(self):
        """
        Mapping of words to their integer representations.
        """
        self.words_int = dict((t,i) for i,t in enumerate(self.words))


    def view_contexts(self, ctx_type, as_strings=False, as_slices=False):
        """
        Displays a tokenization of the corpus.

        Parameters
        ----------
        ctx_type : string-like
           The type of a tokenization.
        as_strings : Boolean, optional
            If True, string representations of words are returned.
            Otherwise, integer representations are returned. Default
            is `False`.
	as_slices : Boolean, optional
            If True, a list of slices corresponding to 'ctx_type' is 
	    returned. Otherwise, integer representations are returned.
	    Default is `False`.

        Returns
        -------
        A tokenized view of `corpus`.

        See Also
        --------
        Corpus
        BaseCorpus
        """
        token_list = super(Corpus, self).view_contexts(ctx_type)
	 
        if as_strings:
            token_list_ = []
            for token in token_list:
                token = self.words[token]
                token_list_.append(token)

            return token_list_

	if as_slices:
	    meta_list = super(Corpus, self).view_metadata(ctx_type)
	    indices = meta_list['idx'] 

	    slices = []
	    slices.append(slice(0, indices[0]))
	    for i in xrange(len(indices) - 1):
		slices.append(slice(indices[i], indices[i+1]))

	    return slices	    

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
        c.words = arrays_in['words']
        c.context_types = arrays_in['context_types'].tolist()

        c.context_data = list()
        for n in c.context_types:
            t = arrays_in['context_data_' + n]
            c.context_data.append(t)

        c.__set_words_int()

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
        arrays_out['words'] = self.words
        arrays_out['context_types'] = np.asarray(self.context_types)

        for i,t in enumerate(self.context_data):
            key = 'context_data_' + self.context_types[i]
            arrays_out[key] = t

        np.savez(file, **arrays_out)


    def apply_stoplist(self, stoplist=[], freq=0):
        """ 
        Takes a Corpus object and returns a copy of it with words in the
        stoplist removed and with words of frequency <= `freq` removed.

        Parameters
        ----------
	stoplist : list
	    The list of words to be removed.
	freq : integer, optional
	    A threshold where words of frequency <= 'freq' are removed. 
	    Default is 0.
            
        Returns
        -------
	Copy of corpus with words in the stoplist and words of frequnecy
	<= 'freq' removed.

        See Also
        --------
        Corpus
        """
        
	if freq:
            #TODO: Use the TF model instead

            print 'Computing collection frequencies'
            cfs = np.zeros_like(self.words, dtype=self.corpus.dtype)
    
            for word in self.corpus:
                cfs[word] += 1

            print 'Selecting words of frequency <=', freq
            freq_stop = np.arange(cfs.size)[(cfs <= freq)]
            stop = set(freq_stop)
        else:
            stop = set()

        for t in stoplist:
            if t in self.words:
                stop.add(self.words_int[t])

        if not stop:
            print 'Stop list is empty.'
            return self
    
        print 'Removing stop words'
        f = np.vectorize(lambda x: x not in stop)
        corpus = self.corpus[f(self.corpus)]

        print 'Rebuilding corpus'
        corpus = [self.words[i] for i in corpus]
        context_data = []
        for i in xrange(len(self.context_data)):
            print 'Recomputing token breaks:', self.context_types[i]
            tokens = self.view_contexts(self.context_types[i])
            spans = [t[f(t)].size for t in tokens]
            tok = self.context_data[i].copy()
            tok['idx'] = np.cumsum(spans)
            context_data.append(tok)

        return Corpus(corpus, context_data=context_data, context_types=self.context_types)



#
# Testing
#


    
def test_file():

    from vsm.util.corpustools import random_corpus

    c = random_corpus(10000, 500, 1, 20, context_type='foo', metadata=True)

    from tempfile import NamedTemporaryFile
    import os

    try:
        tmp = NamedTemporaryFile(delete=False, suffix='.npz')
        c.save(tmp.name)
        tmp.close()
        c_reloaded = c.load(tmp.name)

        assert (c.corpus == c_reloaded.corpus).all()
        assert (c.words == c_reloaded.words).all()
        assert c.words_int == c_reloaded.words_int
        assert c.context_types == c_reloaded.context_types
        for i in xrange(len(c.context_data)):
            assert (c.context_data[i] == c_reloaded.context_data[i]).all()
    
    finally:
        os.remove(tmp.name)

    return c_reloaded
