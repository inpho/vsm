import os

import numpy as np
from unidecode import unidecode
from codecs import open

from vsm.corpus import Corpus
from util import *

from progressbar import ProgressBar, Percentage, Bar

__all__ = ['empty_corpus', 'random_corpus',
           'toy_corpus', 'corpus_fromlist',
           'file_corpus', 'dir_corpus', 'coll_corpus', 'json_corpus',
           'corpus_from_strings', 'walk_corpus']



def corpus_from_strings(strings, metadata=[], decode=False,
                        nltk_stop=True, stop_freq=0, add_stop=None, tokenizer=word_tokenize):
    """
    Takes a list of strings and returns a Corpus object whose document
    tokens are the strings.
    :param tokenizer: word tokenization function. Defaults to `vsm.extensions.corpusbuilders.util.word_tokenize`.
    :type tokenizer: lambda s -> tokens

    """
    if decode:
        for i in xrange(len(strings)):
            if isinstance(strings[i], unicode):
                strings[i] = unidecode(strings[i])

    documents = [tokenizer(s) for s in strings]
    corpus = sum(documents, [])
    indices = np.cumsum([len(d) for d in documents])
    del documents

    if len(metadata) == 0:
        metadata = ['document_{0}'.format(i) for i in xrange(len(strings))]
    md_type = np.array(metadata).dtype
    dtype = [('idx', np.int), ('document_label', md_type)]
    context_data = [np.array(zip(indices, metadata), dtype=dtype)]

    c = Corpus(corpus, context_data=context_data, context_types=['document'])
    if nltk_stop or stop_freq or add_stop:
        c = apply_stoplist(c, nltk_stop=nltk_stop,
                              freq=stop_freq, add_stop=add_stop)
    return c



def empty_corpus(context_type='document'):
    """
    Creates an empty Corpus with defined context_type.

    :param context_type: A type of tokenization. Default is 'document'.
    :type context_type: string

    :returns: An empty Corpus with no words or context_data.

    :See Also: :class:`vsm.corpus.Corpus`
    """
    return Corpus([],
                  context_data=[np.array([], dtype=[('idx', np.int)])],
                  context_types=[context_type])


def random_corpus(corpus_len,
                  n_words,
                  min_token_len,
                  max_token_len,
                  context_type='document',
                  metadata=False,
                  seed=None):
    """
    Generates a random integer corpus.

    :param corpus_len: Size of the Corpus.
    :type corpus_len: int

    :param n_words: Number of words to draw random integers from.
    :type n_words: int

    :param min_token_len: minimum token length used to create indices
        for corpus.
    :type min_token_len: int

    :param max_token_len: maximum token length used to create indices
        for corpus.
    :type max_token_len: int

    :param context_type: A type of tokenization. Default is 'document'.
    :type context_type: string, optional

    :param metadata: If `True` generates metadata. If `False` the only
        metadata for the corpus is the index information.
    :type metadata: boolean, optional
    
    :param tokenizer: word tokenization function. Defaults to `vsm.extensions.corpusbuilders.util.word_tokenize`.
    :type tokenizer: lambda s -> tokens

    :returns: Corpus object with random integers as its entries. 

    :See Also: :class:`vsm.corpus.Corpus`
    """
    random_state = np.random.RandomState(seed)
    corpus = random_state.randint(n_words, size=corpus_len)

    indices = []
    i = np.random.randint(min_token_len, max_token_len)
    while i < corpus_len:
        indices.append(i)
        i += np.random.randint(min_token_len, max_token_len)
    indices.append(corpus_len)

    if metadata:
        metadata_ = ['{0}_{1}'.format(context_type, i)
                     for i in xrange(len(indices))]
        dtype=[('idx', np.array(indices).dtype), 
               (context_type + '_label', np.array(metadata_).dtype)]
        rand_tok = np.array(zip(indices, metadata_), dtype=dtype)
    else:
        rand_tok = np.array([(i,) for i in indices], 
                            dtype=[('idx', np.array(indices).dtype)])

    return Corpus(corpus, context_types=[context_type], context_data=[rand_tok])


def corpus_fromlist(ls, context_type='context'):
    """
    Takes a list of lists or arrays containing strings or integers and
    returns a Corpus object. The label associated to a given context
    is `context_type` prepended to the context index.
    
    :param ls: List of lists or List of arrays containing strings or integers.
    :type ls: list

    :param context_type: A type of tokenization.
    :type context_type: string, optional

    :returns: A Corpus object built from `ls`.

    :See Also: :class:`vsm.corpus.Corpus`

    **Examples**

    >>> ls = [['a', 'b'], ['c'], ['d', 'e']]
    >>> c = corpus_fromlist(ls, context_type='sentence')
    >>> c.view_contexts('sentence', as_strings=True)
    [array(['a', 'b'], dtype='|S1'),
     array(['c'], dtype='|S1'),
     array(['d', 'e'], dtype='|S1')]
    >>> c.context_data
    [array([(2, 'sentence_0'), (3, 'sentence_1'), (5, 'sentence_2')], 
          dtype=[('idx', '<i8'), ('sentence_label', '|S10')])]
    """
    corpus = [w for ctx in ls for w in ctx]

    indices = np.cumsum([len(sbls) for sbls in ls])
    metadata = ['{0}_{1}'.format(context_type, i)
                for i in xrange(len(indices))]
    md_type = np.array(metadata).dtype
    dtype = [('idx', np.int), (context_type + '_label', md_type)]
    context_data = [np.array(zip(indices, metadata), dtype=dtype)]

    return Corpus(corpus, context_data=context_data,
                  context_types=[context_type])


def toy_corpus(plain_corpus, is_filename=False, encoding='utf8', nltk_stop=False,
               stop_freq=0, add_stop=None, decode=False,
               metadata=None, autolabel=False, tokenizer=word_tokenize, simple=False):
    """
    `toy_corpus` is a convenience function for generating Corpus
    objects from a given string or a single file.

    `toy_corpus` will perform both word and document-level
    tokenization. It will also strip punctuation and arabic numerals
    outside the range 1-29. All letters are made lowercase.

    Document tokens are delimited by two or more line breaks. E.g.,

        <document 0>

        <document 1>

        ...

        <document n>

    where <document i> is any chunk of text to be tokenized by word.

    :param plain_corpus: String containing a plain-text corpus or a 
        filename of a file containing one.
    :type plain_corpus: string-like
    
    :param is_filename: If `True` then `plain_corpus` is treated like
        a filename. Otherwise, `plain_corpus` is presumed to contain 
        the corpus. Default is `False`.
    :type is_filename: boolean, optional

    :param encoding: A string indicating the file encoding or 'detect',
        in which case `chardet` is used to automatically guess the encoding.
        Default is `utf8`.
    :type encoding: string, optional
    
    :param nltk_stop: If `True` then the corpus object is masked using
        the NLTK English stop words. Default is `False`.
    :type nltk_stop: boolean, optional

    :param stop_freq: The upper bound for a word to be masked on the basis of its
        collection frequency. Default is 0.
    :type stop_freq: int, optional

    :param add_stop: A list of stop words. Default is `None`.
    :type add_stop: array-like, optional
    
    :param decode: If `True` then unicode characters are converted to
        ASCII. Default is `False`.
    :type decode: boolean, optional

    :param metadata: A list of strings providing metadata about the documents. If
        provided, must have length equal to the number of documents.
        Default is `None`.
    :type metadata: array-like, optional
    
    :param autolabel: A boolean specifying whether to automatically label
        documents by position in file. Default is False
    :type metadata: boolean, optional
    
    :param tokenizer: word tokenization function. Defaults to `vsm.extensions.corpusbuilders.util.word_tokenize`.
    :type tokenizer: lambda s -> tokens
    
    :returns: c : a Corpus object
        Contains the tokenized corpus built from the input plain-text
        corpus. Document tokens are named `documents`.

    :See Also: :class:`vsm.corpus.Corpus`, 
        :meth:`vsm.corpus.util.paragraph_tokenize`, 
        :meth:`vsm.corpus.util.apply_stoplist`
    """
    if is_filename:
        if encoding == 'detect':
            encoding = detect_encoding(plain_corpus)

        with open(plain_corpus, 'rb', encoding=encoding) as f:
            plain_corpus = f.read()

    if decode:
        plain_corpus = unidecode(plain_corpus)

    docs = paragraph_tokenize(plain_corpus)
    docs = [tokenizer(d) for d in docs]

    corpus = sum(docs, [])
    tok = np.cumsum(np.array([len(d) for d in docs]))

    if not metadata and autolabel:
        metadata = ['Document {0}'.format(i) for i in range(len(tok))]

    if metadata:
        if not len(metadata) == len(tok):
            msg = 'Metadata mismatch: metadata length is {0} and number'\
                   'of documents is {1}'.format(len(metadata), len(tok))
            raise Exception(msg)
        else:
            dtype = [('idx', np.array(tok).dtype),
                     ('document_label', np.array(metadata).dtype)]
            tok = np.array(zip(tok, metadata), dtype=dtype)
    else:
        dtype = [('idx', np.array(tok).dtype)]
        tok = np.array([(i,) for i in tok], dtype=dtype)
    
    c = Corpus(corpus, context_data=[tok], context_types=['document'])
    if nltk_stop or stop_freq or add_stop:
        c = apply_stoplist(c, nltk_stop=nltk_stop,
                              freq=stop_freq, add_stop=add_stop)
    return c

def file_tokenize(text, tokenizer=word_tokenize, simple=False):
    """
    `file_tokenize` is a helper function for :meth:`file_corpus`.
    
    Takes a string that is content in a file and returns words
    and corpus data.

    :param text: Content in a plain text file.
    :type text: string
    
    :param tokenizer: word tokenization function. Defaults to `vsm.extensions.corpusbuilders.util.word_tokenize`.
    :type tokenizer: lambda s -> tokens

    :returns: words : List of words.
        Words in the `text` tokenized by :meth:`vsm.corpus.util.word_tokenize`.
        corpus_data : Dictionary with context type as keys and
        corresponding tokenizations as values. The tokenizations
        are np.arrays.
    """
    words, par_tokens, sent_tokens = [], [], []
    par_break, sent_break, par_n, sent_n = 0, 0, 0, 0

    pars = paragraph_tokenize(text)

    for par in pars:
        if simple == True:
            w = tokenizer(par)
            words.extend(w)
            par_break += len(w)
            par_tokens.append((par_break, par_n))
        else:
            sents = sentence_tokenize(par)
            for sent in sents:
                w = tokenizer(sent)
                words.extend(w)
                sent_break += len(w)
                sent_tokens.append((sent_break, par_n, sent_n))
                sent_n += 1
            par_tokens.append((sent_break, par_n))
        par_n += 1

    idx_dt = ('idx', np.int32)
    if simple == False:
        sent_label_dt = ('sentence_label', np.array(sent_n, np.str_).dtype)
    par_label_dt = ('paragraph_label', np.array(par_n, np.str_).dtype)

    corpus_data = dict()
    dtype = [idx_dt, par_label_dt]
    corpus_data['paragraph'] = np.array(par_tokens, dtype=dtype)
    if simple == False:
        dtype = [idx_dt, par_label_dt, sent_label_dt]
        corpus_data['sentence'] = np.array(sent_tokens, dtype=dtype)

    return words, corpus_data


def file_corpus(filename, encoding='utf8', nltk_stop=True, stop_freq=1, 
                add_stop=None, decode=False, simple=False,
                tokenizer=word_tokenize):
    """
    `file_corpus` is a convenience function for generating Corpus
    objects from a a plain text corpus contained in a single string.

    `file_corpus` will strip punctuation and arabic numerals outside
    the range 1-29. All letters are made lowercase.

    :param filename: File name of the plain text file.
    :type plain_dir: string-like

    :param encoding: A string indicating the file encoding or 'detect',
        in which case `chardet` is used to automatically guess the encoding.
        Default is `utf8`.
    :type encoding: string, optional
    
    :param nltk_stop: If `True` then the corpus object is masked 
        using the NLTK English stop words. Default is `False`.
    :type nltk_stop: boolean, optional
    
    :param stop_freq: The upper bound for a word to be masked on 
        the basis of its collection frequency. Default is 1.
    :type stop_freq: int, optional
    
    :param add_stop: A list of stop words. Default is `None`.
    :type add_stop: array-like, optional
    
    :param decode: If `True` then unicode characters are converted to
        ASCII. Default is `False`.
    :type decode: boolean, optional

    :returns: c : a Corpus object
        Contains the tokenized corpus built from the input plain-text
        corpus. Document tokens are named `documents`.
    
    :See Also: :class:`vsm.corpus.Corpus`, 
        :meth:`file_tokenize`, 
        :meth:`vsm.corpus.util.apply_stoplist`
    """
    if encoding == 'detect':
        encoding = detect_encoding(filename)
    try:
        with open(filename, mode='r', encoding=encoding) as f:
            text = f.read()
    except UnicodeDecodeError: 
        encoding = detect_encoding(filename)

    if decode:
        text = unidecode(text)

    words, tok = file_tokenize(text, simple=simple, tokenizer=tokenizer)
    names, data = zip(*tok.items())
    
    c = Corpus(words, context_data=data, context_types=names)
    if nltk_stop or stop_freq or add_stop:
        c = apply_stoplist(c, nltk_stop=nltk_stop,
                              freq=stop_freq, add_stop=add_stop)
    return c


def json_corpus(json_file, doc_key, label_key, encoding='utf8',
                nltk_stop=False, stop_freq=0, add_stop=None, tokenizer=word_tokenize):
    """
    `json_corpus` is a convenience function for generating Corpus
    objects from a json file. It construct a corpus, document labels
    and metadata respectively from the specified fields in the json file.

    `json_corpus` will perform word-level tokenization. 
    It will also strip punctuation and arabic numerals
    outside the range 1-29. All letters are made lowercase.

    :param json_file: Json file name containing documents and metadata.
    :type json_file: string-like
    
    :param doc_key: Name of the key for documents.
    :type doc_key: string-like

    :param label_key: Name of the key used for document labels. Labels are 
    used when a viewer function outputs a list of documents. Any field other
    than `doc_key` and `label_key` is stored as metadata.
    :type label_key: string-like

    :param encoding: A string indicating the file encoding or 'detect',
        in which case `chardet` is used to automatically guess the encoding.
        Default is `utf8`.
    :type encoding: string, optional
    
    :param nltk_stop: If `True` then the corpus object is masked using
        the NLTK English stop words. Default is `False`.
    :type nltk_stop: boolean, optional

    :param stop_freq: The upper bound for a word to be masked on the basis of its
        collection frequency. Default is 0.
    :type stop_freq: int, optional

    :param add_stop: A list of stop words. Default is `None`.
    :type add_stop: array-like, optional
    
    :param tokenizer: word tokenization function. Defaults to `vsm.extensions.corpusbuilders.util.word_tokenize`.
    :type tokenizer: lambda s -> tokens

    :returns: c : a Corpus object
        Contains the tokenized corpus built from the input plain-text
        corpus. Document tokens are named `documents`.

    :See Also: :class:`vsm.corpus.Corpus`, 
        :meth:`vsm.corpus.util.paragraph_tokenize`, 
        :meth:`vsm.corpus.util.apply_stoplist`
    """
    import json

    if encoding == 'detect':
        encoding = detect_encoding(json_file)
    with open(json_file, 'r', encoding=encoding) as f:
        json_data = json.load(f)

    docs = []
    label = []
    metadata = []
    for i in json_data:
        docs.append(i.pop(doc_key, None).encode('ascii','ignore'))
        label.append(i.pop(label_key, None))
        metadata.append(i)   # metadata are all the rest

    docs = [tokenizer(d) for d in docs]

    corpus = sum(docs, [])
    tok = np.cumsum(np.array([len(d) for d in docs]))

    # add document label and metadata
    dtype = [('idx', np.array(tok).dtype),
             ('document_label', np.array(label).dtype),
             ('metadata', np.array(metadata).dtype)]         # todo: create separate dtype for each key?
    tok = np.array(zip(tok, label, metadata), dtype=dtype)

    
    c = Corpus(corpus, context_data=[tok], context_types=['document'])
    if nltk_stop or stop_freq or add_stop:
        c = apply_stoplist(c, nltk_stop=nltk_stop,
                              freq=stop_freq, add_stop=add_stop)
    return c



def dir_tokenize(chunks, labels, chunk_name='article', paragraphs=True,
                 verbose=1, tokenizer=word_tokenize, simple=False):
    """`dir_tokenize` is a helper function for :meth:`dir_corpus`.

    Takes a list of strings and labels and returns words and corpus data.

    :param chunks: List of strings.
    :type chunks: list

    :param labels: List of strings.
    :type labels: list

    :param chunk_name: The name of the tokenization corresponding to
        individual files. For example, if the input strings are pages
        of a book, one might set `chunk_name` to `pages`. Default is
        `articles`.  :type chunk_name: string-like, optional
    
    :param paragraphs: If `True`, a paragraph-level tokenization 
        is included. Defaults to `True`.
    :type paragraphs: boolean, optional

    :param verbose: Verbosity level. 1 prints a progress bar.
    :type verbose: int, default 1
    
    :param tokenizer: word tokenization function. Defaults to `vsm.extensions.corpusbuilders.util.word_tokenize`.
    :type tokenizer: lambda s -> tokens
    
    :returns: words : List of words.
        words in the `chunks` tokenized by :meth: word_tokenize.
        corpus_data : Dictionary with context type as keys and
        corresponding tokenizations as values. The tokenizations
        are np.arrays.

    """
    words, chk_tokens, sent_tokens = [], [], []
    chk_break, par_break, sent_break = 0, 0 ,0
    chk_n, par_n, sent_n = 0, 0, 0
    
    if verbose == 1:
        pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(chunks)).start()

    if simple:
        paragraphs = False;

    if paragraphs:
        par_tokens = []
        par_n = 0
        
        for chk, label in zip(chunks, labels):
            # print 'Tokenizing', label
            pars = paragraph_tokenize(chk)

            for par in pars:
                sents = sentence_tokenize(par)
                
                for sent in sents:
                    w = tokenizer(sent)
                    words.extend(w)
                    sent_break += len(w)
                    sent_tokens.append((sent_break, label, par_n, sent_n))
                    sent_n += 1

                par_tokens.append((sent_break, label, par_n))
                par_n += 1
    
            if verbose == 1:
                pbar.update(chk_n)

            chk_tokens.append((sent_break, label))

            chk_n += 1
    else:
        for chk, label in zip(chunks, labels):
            print 'Tokenizing', label
            if simple:
                w = tokenizer(chk)
                words.extend(w)
                chk_break += len(w)
                chk_tokens.append((chk_break, label))
  

            else:
                sents = sentence_tokenize(chk)

                for sent in sents:
                    w = tokenizer(sent)
                    words.extend(w)
                    sent_break += len(w)
                    sent_tokens.append((sent_break, label, sent_n))
                    sent_n += 1
                chk_tokens.append((sent_break, label))
    
            if verbose == 1:
                pbar.update(chk_n)

  
            chk_n += 1


    idx_dt = ('idx', np.int32)
    label_dt = (chunk_name + '_label', np.array(labels).dtype)
    if not simple:
        sent_label_dt = ('sentence_label', np.array(sent_n, np.str_).dtype)
    corpus_data = dict()
    dtype = [idx_dt, label_dt]
    corpus_data[chunk_name] = np.array(chk_tokens, dtype=dtype)

    if paragraphs:
        par_label_dt = ('paragraph_label', np.array(par_n, np.str_).dtype)
        dtype = [idx_dt, label_dt, par_label_dt]
        corpus_data['paragraph'] = np.array(par_tokens, dtype=dtype)
        dtype = [idx_dt, label_dt, par_label_dt, sent_label_dt]
        corpus_data['sentence'] = np.array(sent_tokens, dtype=dtype)
    elif not simple:
        dtype = [idx_dt, label_dt, sent_label_dt]
        corpus_data['sentence'] = np.array(sent_tokens, dtype=dtype)
    
    if verbose == 1:
        pbar.finish()

    return words, corpus_data



def dir_corpus(plain_dir, chunk_name='article', encoding='utf8', 
               paragraphs=True, ignore=['.json','.log','.pickle'], 
               nltk_stop=True, stop_freq=1, add_stop=None, decode=False, 
               verbose=1, simple=False, tokenizer=word_tokenize):
    """
    `dir_corpus` is a convenience function for generating Corpus
    objects from a directory of plain text files.

    `dir_corpus` will retain file-level tokenization and perform
    sentence and word tokenizations. Optionally, it will provide
    paragraph-level tokenizations.

    It will also strip punctuation and arabic numerals outside the
    range 1-29. All letters are made lowercase.

    :param plain_dir: String containing directory containing a 
        plain-text corpus.
    :type plain_dir: string-like
    
    :param chunk_name: The name of the tokenization corresponding 
        to individual files. For example, if the files are pages 
        of a book, one might set `chunk_name` to `pages`. Default 
        is `articles`.
    :type chunk_name: string-like, optional
    
    :param encoding: A string indicating the file encoding or 'detect',
        in which case `chardet` is used to automatically guess the encoding.
        Default is `utf8`.
    :type encoding: string, optional
    
    :param paragraphs: If `True`, a paragraph-level tokenization 
        is included. Defaults to `True`.
    :type paragraphs: boolean, optional
    
    :param ignore: The list containing suffixes of files to be filtered.
        The suffix strings are normally file types. Default is ['.json',
        '.log','.pickle'].
    :type ignore: list of strings, optional

    :param nltk_stop: If `True` then the corpus object is masked 
        using the NLTK English stop words. Default is `False`.
    :type nltk_stop: boolean, optional
    
    :param stop_freq: The upper bound for a word to be masked on 
        the basis of its collection frequency. Default is 1.
    :type stop_freq: int, optional
    
    :param add_stop: A list of stop words. Default is `None`.
    :type add_stop: array-like, optional
    
    :param decode: If `True` then unicode characters are converted to
        ASCII. Default is `False`.
    :type decode: boolean, optional

    :param verbose: Verbosity level. 1 prints a progress bar.
    :type verbose: int, default 1 

    :returns: c : a Corpus object
        Contains the tokenized corpus built from the input plain-text
        corpus. Document tokens are named `documents`.
    
    :See Also: :class:`vsm.corpus.Corpus`, 
            :meth:`dir_tokenize`, 
            :meth:`vsm.corpus.util.apply_stoplist`
    """
    chunks = []
    filenames = os.listdir(plain_dir)
    filenames = filter_by_suffix(filenames, ignore)
    filenames.sort()

    for filename in filenames:
        filename = os.path.join(plain_dir, filename)
        if encoding == 'detect':
            encoding = detect_encoding(filename)
        try:
            if decode:
                with open(filename, mode='r', encoding=encoding) as f:
                    chunks.append(unidecode(f.read()))
            else:
                with open(filename, mode='r', encoding=encoding) as f:
                    chunks.append(f.read())
        except UnicodeDecodeError:
            encoding = detect_encoding(filename)
            if decode:
                with open(filename, mode='r', encoding=encoding) as f:
                    chunks.append(unidecode(f.read()))
            else:
                with open(filename, mode='r', encoding=encoding) as f:
                    chunks.append(f.read())

    words, tok = dir_tokenize(chunks, filenames, chunk_name=chunk_name,
                              paragraphs=paragraphs, verbose=verbose,
                              simple=simple, tokenizer=tokenizer)
    names, data = zip(*tok.items())
    
    c = Corpus(words, context_data=data, context_types=names)
    if nltk_stop or stop_freq or add_stop:
        c = apply_stoplist(c, nltk_stop=nltk_stop,
                              freq=stop_freq, add_stop=add_stop)
    return c


def coll_tokenize(books, book_names, verbose=1, tokenizer=word_tokenize, simple=False):
    """
    `coll_tokenize` is a helper function for :meth:`coll_corpus`.

    Takes a list of books and `book_names`, and returns words 
    and corpus data.

    :param books: List of books.
    :type books: list

    :param book_names: List of book names.
    :type book_names: list

    :param verbose: Verbosity level. 1 prints a progress bar.
    :type verbose: int, default 1
    
    :param tokenizer: word tokenization function. Defaults to `vsm.extensions.corpusbuilders.util.word_tokenize`.
    :type tokenizer: lambda s -> tokens

    :returns: words : List of words.
        words in the `books` tokenized by :meth:`word_tokenize`.
        corpus_data : Dictionary with context type as keys and
        corresponding tokenizations as values. The tokenizations
        are np.arrays.
    """
    words, book_tokens, page_tokens, sent_tokens = [], [], [], []
    page_break, sent_break, book_n, page_n, sent_n = 0, 0, 0, 0, 0

    if verbose == 1:
        pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(books)).start()

    for book, book_label in zip(books, book_names):
        # print 'Tokenizing', book_label
        for page, page_file in book:
            if simple:
                w = tokenizer(page)
                words.extend(w)
                page_break += len(w)
                page_tokens.append((page_break, page_n, book_label, page_file))
            else:
                sents = sentence_tokenize(page)

                for sent in sents:
                    w = tokenizer(sent)
                    words.extend(w)
                    sent_break += len(w)
                    sent_tokens.append((sent_break, sent_n,
                                        page_n, book_label, page_file))
                    sent_n += 1

                page_tokens.append((sent_break, page_n, book_label, page_file))

            page_n += 1

        if verbose == 1:
            pbar.update(book_n)
            
        if not simple:
            book_tokens.append((sent_break, book_label))
        else:
            book_tokens.append((page_break, book_label))
        book_n += 1

    idx_dt = ('idx', np.int32)
    book_label_dt = ('book_label', np.array(book_names).dtype)
    page_label_dt = ('page_label', np.array(page_n, np.str_).dtype)
    if not simple:
        sent_label_dt = ('sentence_label', np.array(sent_n, np.str_).dtype)
    files = [f for (a,b,c,f) in page_tokens]
    file_dt = ('file', np.array(files, np.unicode_).dtype)

    corpus_data = dict()
    dtype = [idx_dt, book_label_dt]
    corpus_data['book'] = np.array(book_tokens, dtype=dtype)
    dtype = [idx_dt, page_label_dt, book_label_dt, file_dt]
    corpus_data['page'] = np.array(page_tokens, dtype=dtype)
    if not simple:
        dtype = [idx_dt, sent_label_dt, page_label_dt, book_label_dt, file_dt]
        corpus_data['sentence'] = np.array(sent_tokens, dtype=dtype)

    if verbose == 1:
        pbar.finish()

    return words, corpus_data

#TODO: This should be a whitelist not a blacklist
def coll_corpus(coll_dir, encoding='utf8', ignore=['.json', '.log', '.pickle'],
                nltk_stop=True, stop_freq=1, add_stop=None, 
                decode=False, verbose=1, simple=False, tokenizer=word_tokenize):
    """
    `coll_corpus` is a convenience function for generating Corpus
    objects from a directory of plain text files.

    It will also strip punctuation and arabic numerals outside the
    range 1-29. All letters are made lowercase.

    :param coll_dir: Directory containing a collections of books
        which contain pages as plain-text files.
    :type coll_dir: string-like
    
    :param encoding: A string indicating the file encoding or 'detect',
        in which case `chardet` is used to automatically guess the encoding.
        Default is `utf8`.
    :type encoding: string, optional
    
    :param ignore: The list containing suffixes of files to be filtered.
        The suffix strings are normally file types. Default is ['.json',
        '.log','.pickle'].
    :type ignore: list of strings, optional

    :param nltk_stop: If `True` then the corpus object is masked 
        using the NLTK English stop words. Default is `False`.
    :type nltk_stop: boolean, optional
    
    :param stop_freq: The upper bound for a word to be masked on 
        the basis of its collection frequency. Default is 1.
    :type stop_freq: int, optional
    
    :param add_stop: A list of stop words. Default is `None`.
    :type add_stop: array-like, optional
    
    :param decode: If `True` then unicode characters are converted to
        ASCII. Default is `False`.
    :type decode: boolean, optional

    :param verbose: Verbosity level. 1 prints a progress bar.
    :type verbose: int, default 1 

    :returns: c : a Corpus object
        Contains the tokenized corpus built from the plain-text files
        in `coll_dir` corpus. Document tokens are named `documents`.
    """
    books = []
    book_names = os.listdir(coll_dir)
    book_names = filter_by_suffix(book_names, ignore)
    book_names.sort()

    for book_name in book_names:
        pages = []
        book_path = os.path.join(coll_dir, book_name)
        page_names = os.listdir(book_path)
        page_names = filter_by_suffix(page_names, ignore)
        page_names.sort()

        for page_name in page_names:
            page_file = book_name + '/' + page_name
            page_name = os.path.join(book_path, page_name)
            if encoding == 'detect':
                encoding = detect_encoding(page_name)
            try:
                if decode:
                    with open(page_name, mode='r', encoding=encoding) as f:
                        pages.append((unidecode(f.read()), page_file))
                else:
                    with open(page_name, mode='r', encoding=encoding) as f:
                        pages.append((f.read(), page_file))
            except UnicodeDecodeError:
                encoding = detect_encoding(page_name)
                if decode:
                    with open(page_name, mode='r', encoding=encoding) as f:
                        pages.append((unidecode(f.read()), page_file))
                else:
                    with open(page_name, mode='r', encoding=encoding) as f:
                        pages.append((f.read(), page_file))


        books.append(pages)

    words, tok = coll_tokenize(books, book_names, simple=simple,
                               tokenizer=tokenizer)
    names, data = zip(*tok.items())
    
    c = Corpus(words, context_data=data, context_types=names)
    in_place_stoplist(c, nltk_stop=nltk_stop,
                       freq=stop_freq, add_stop=add_stop)

    return c


def record_tokenize(records, record_names, verbose=1):
    """
    `record_tokenize` is a helper function for :meth:`record_corpus`.

    Takes a list of books and `book_names`, and returns words 
    and corpus data.

    :param books: List of books.
    :type books: list

    :param book_names: List of book names.
    :type book_names: list

    :param verbose: Verbosity level. 1 prints a progress bar.
    :type verbose: int, default 1 

    :returns: words : List of words.
        words in the `books` tokenized by :meth:`word_tokenize`.
        corpus_data : Dictionary with context type as keys and
        corresponding tokenizations as values. The tokenizations
        are np.arrays.
    """
    words, record_tokens, book_tokens, page_tokens, sent_tokens = [], [], [], [], []
    sent_break, record_n, book_n, page_n, sent_n = 0, 0, 0, 0, 0

    if verbose == 1:
        pbar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(books)).start()

    for record, record_label, book_labels in zip(records, record_names, record_book_names):
        # TODO extract book data

        for book, book_label in zip(record, book_labels):
            # print 'Tokenizing', book_label
            for page, page_file in book:
                sents = sentence_tokenize(page)

                for sent in sents:
                    w = word_tokenize(sent)
                    words.extend(w)
                    sent_break += len(w)
                    sent_tokens.append((sent_break, sent_n, page_n, book_label, 
                                        record_label, page_file))
                    sent_n += 1
    
                page_tokens.append((sent_break, page_n, book_label,
                                    record_label, page_file))
                page_n += 1

    
            if verbose == 1:
                pbar.update(book_n)
                
            book_tokens.append((sent_break, book_label, record_label))
            book_n += 1

        record_tokens.append((sent_break, record_label))
        record_n += 1

    idx_dt = ('idx', np.int32)
    record_label_dt = ('record_label', np.array(record_names).dtype)
    book_label_dt = ('book_label', np.array(book_names).dtype)
    page_label_dt = ('page_label', np.array(page_n, np.str_).dtype)
    sent_label_dt = ('sentence_label', np.array(sent_n, np.str_).dtype)
    files = [f for (a,b,c,f) in page_tokens]
    file_dt = ('file', np.array(files, np.str_).dtype)

    corpus_data = dict()
    dtype = [idx_dt, record_label_dt]
    corpus_data['record'] = np.array(record_tokens, dtype=dtype)
    dtype = [idx_dt, book_label_dt, record_label_dt]
    corpus_data['book'] = np.array(book_tokens, dtype=dtype)
    dtype = [idx_dt, page_label_dt, book_label_dt, record_label_dt, file_dt]
    corpus_data['page'] = np.array(page_tokens, dtype=dtype)
    dtype = [idx_dt, sent_label_dt, page_label_dt, book_label_dt, record_label_dt, file_dt]
    corpus_data['sentence'] = np.array(sent_tokens, dtype=dtype)

    if verbose == 1:
        pbar.finish()

    return words, corpus_data


#TODO: This should be a whitelist not a blacklist
def record_corpus(base_dir, encoding='utf8', ignore=['.json', '.log', '.pickle'],
                nltk_stop=True, stop_freq=1, add_stop=None, 
                decode=False, verbose=1):
    """
    `record_corpus` is a convenience function for generating Corpus
    objects from a directory of plain text files.

    It will also strip punctuation and arabic numerals outside the
    range 1-29. All letters are made lowercase.

    :param base_dir: Directory containing a recordections of books
        which contain pages as plain-text files.
    :type base_dir: string-like
    
    :param encoding: A string indicating the file encoding or 'detect',
        in which case `chardet` is used to automatically guess the encoding.
        Default is `utf8`.
    :type encoding: string, optional
    
    :param ignore: The list containing suffixes of files to be filtered.
        The suffix strings are normally file types. Default is ['.json',
        '.log','.pickle'].
    :type ignore: list of strings, optional

    :param nltk_stop: If `True` then the corpus object is masked 
        using the NLTK English stop words. Default is `False`.
    :type nltk_stop: boolean, optional
    
    :param stop_freq: The upper bound for a word to be masked on 
        the basis of its recordection frequency. Default is 1.
    :type stop_freq: int, optional
    
    :param add_stop: A list of stop words. Default is `None`.
    :type add_stop: array-like, optional
    
    :param decode: If `True` then unicode characters are converted to
        ASCII. Default is `False`.
    :type decode: boolean, optional

    :param verbose: Verbosity level. 1 prints a progress bar.
    :type verbose: int, default 1 

    :returns: c : a Corpus object
        Contains the tokenized corpus built from the plain-text files
        in `record_dir` corpus. Document tokens are named `documents`.
    """

    records = []
    record_names = os.listdir(base_dir)
    record_names = filter_by_suffix(record_names, ignore)
    record_names.sort()
    record_book_names = []

    for record_dir in record_names:
        books = []
        book_names = os.listdir(record_dir)
        book_names = filter_by_suffix(book_names, ignore)
        book_names.sort()
    
        for book_name in book_names:
            pages = []
            book_path = os.path.join(record_dir, book_name)
            page_names = os.listdir(book_path)
            page_names = filter_by_suffix(page_names, ignore)
            page_names.sort()
    
            for page_name in page_names:
                page_file = book_name + '/' + page_name
                page_name = os.path.join(book_path, page_name)
                if encoding == 'detect':
                    encoding = detect_encoding(page_name)
                if decode:
                    with open(page_name, mode='r', encoding=encoding) as f:
                        pages.append((unidecode(f.read()), page_file))
                else:
                    with open(page_name, mode='r', encoding=encoding) as f:
                        pages.append((f.read(), page_file))
    
            books.append(pages)

        record_book_names.append(book_names)
        records.append(books)

    words, tok = record_tokenize(records, record_names, record_book_names)
    names, data = zip(*tok.items())
    
    c = Corpus(words, context_data=data, context_types=names)
    if nltk_stop or stop_freq or add_stop:
        c = apply_stoplist(c, nltk_stop=nltk_stop,
                              freq=stop_freq, add_stop=add_stop)
    return c


def walk_corpus(walk_dir, chunk_name='document', encoding='utf8', 
                ignore=['.json', '.log', '.pickle'],
                nltk_stop=True, stop_freq=1, add_stop=None, 
                decode=False, verbose=1, simple=False, tokenizer=word_tokenize):

    filenames = []
    for root, dirs, files in os.walk(walk_dir):
        for file in files:
            filenames.append(os.path.join(root, file))

    # filter the blacklist (typically .json, .log, etc.)
    filenames = filter_by_suffix(filenames, ignore)
    files = []
    for filename in filenames:
        if encoding == 'detect':
            encoding = detect_encoding(filename)

        try:
            if decode:
                with open(filename, mode='r', encoding=encoding) as f:
                    files.append(unidecode(f.read()))
            else:
                with open(filename, mode='r', encoding=encoding) as f:
                    files.append(f.read())
        except UnicodeDecodeError:
            encoding = detect_encoding(filename)
            if decode:
                with open(filename, mode='r', encoding=encoding) as f:
                    files.append(unidecode(f.read()))
            else:
                with open(filename, mode='r', encoding=encoding) as f:
                    files.append(f.read())

    words, tok = dir_tokenize(files, filenames, chunk_name=chunk_name,
        paragraphs=False, verbose=verbose, simple=simple, tokenizer=tokenizer)
    names, data = zip(*tok.items())

    c = Corpus(words, context_data=data, context_types=names)
    if nltk_stop or stop_freq or add_stop:
        c = apply_stoplist(c, nltk_stop=nltk_stop,
                              freq=stop_freq, add_stop=add_stop)
    return c


###########
# Testing #
###########

def test_toy_corpus():

    keats = ('She dwells with Beauty - Beauty that must die;\n\n'
             'And Joy, whose hand is ever at his lips\n\n' 
             'Bidding adieu; and aching Pleasure nigh,\n\n'
             'Turning to poison while the bee-mouth sips:\n\n'
             'Ay, in the very temple of Delight\n\n'
             'Veil\'d Melancholy has her sovran shrine,\n\n'
             'Though seen of none save him whose strenuous tongue\n\n'
             'Can burst Joy\'s grape against his palate fine;\n\n'
             'His soul shall taste the sadness of her might,\n\n'
             'And be among her cloudy trophies hung.')

    assert toy_corpus(keats)
    assert toy_corpus(keats, nltk_stop=True)
    assert toy_corpus(keats, stop_freq=1)
    assert toy_corpus(keats, add_stop=['and', 'with'])
    assert toy_corpus(keats, nltk_stop=True,
                      stop_freq=1, add_stop=['ay'])

    import os
    from tempfile import NamedTemporaryFile as NFT

    tmp = NFT(delete=False)
    tmp.write(keats)
    tmp.close()

    c = toy_corpus(tmp.name, is_filename=True, 
                   nltk_stop=True, add_stop=['ay'])
    
    assert c
    os.remove(tmp.name)

    return c


def test_dir_tokenize():

    chunks = ['foo foo foo\n\nfoo foo',
             'Foo bar.  Foo bar.', 
             '',
             'foo\n\nfoo']

    labels = [str(i) for i in xrange(len(chunks))]
    words, context_data = dir_tokenize(chunks, labels)

    assert len(words) == 11
    assert len(context_data['article']) == 4
    assert len(context_data['paragraph']) == 6
    assert len(context_data['sentence']) == 7
    
    assert (context_data['article']['idx'] == [5, 9, 9, 11]).all()
    assert (context_data['article']['article_label'] == ['0', '1', '2', '3']).all()
    assert (context_data['paragraph']['idx'] == [3, 5, 9, 9, 10, 11]).all()
    assert (context_data['paragraph']['article_label'] == 
            ['0', '0', '1', '2', '3', '3']).all()
    assert (context_data['paragraph']['par_label'] == 
            ['0', '1', '2', '3', '4', '5']).all()
    assert (context_data['sentence']['idx'] == [3, 5, 7, 9, 9, 10, 11]).all()
    assert (context_data['sentence']['article_label'] == 
            ['0', '0', '1', '1', '2', '3', '3']).all()
    assert (context_data['sentence']['par_label'] == 
            ['0', '1', '2', '2', '3', '4', '5']).all()
    assert (context_data['sentence']['sent_label'] == 
            ['0', '1', '2', '3', '4', '5', '6']).all()


def test_coll_tokenize():

    books = [[('foo foo foo.\n\nfoo foo', '1'),
              ('Foo bar.  Foo bar.', '2')], 
             [('','3'),
              ('foo.\n\nfoo', '4')]]

    book_names = [str(i) for i in xrange(len(books))]
    words, context_data = coll_tokenize(books, book_names)

    assert len(words) == 11
    assert len(context_data['book']) == 2
    assert len(context_data['page']) == 4
    assert len(context_data['sentence']) == 7
    assert (context_data['book']['idx'] == [9, 11]).all()
    assert (context_data['book']['book_label'] == ['0', '1']).all()
    assert (context_data['page']['idx'] == [5, 9, 9, 11]).all()
    assert (context_data['page']['page_label'] == ['0', '1', '2', '3']).all()
    assert (context_data['page']['book_label'] == ['0', '0', '1', '1']).all()
    assert (context_data['sentence']['idx'] == [3, 5, 7, 9, 9, 10, 11]).all()
    assert (context_data['sentence']['sent_label'] == 
            ['0', '1', '2', '3', '4', '5', '6']).all()
    assert (context_data['sentence']['page_label'] == 
            ['0', '0', '1', '1', '2', '3', '3']).all()
    assert (context_data['sentence']['book_label'] == 
            ['0', '0', '0', '0', '1', '1', '1']).all()
    assert (context_data['page']['file'] ==
		['1','2','3','4']).all()
    assert (context_data['sentence']['file'] ==
		['1','1','2','2','3','4','4']).all() 
