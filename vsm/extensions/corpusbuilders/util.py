import re
import string

from chardet.universaldetector import UniversalDetector
import numpy as np

__all__ = ['strip_punc', 'rem_num', 'rehyph',
           'apply_stoplist', 'filter_by_suffix', 'word_tokenize',
           'sentence_tokenize', 'paragraph_tokenize', 'detect_encoding',
           'sentence_span_tokenize', 'in_place_stoplist']


def _unicode_range(start, stop):
    return u''.join([unichr(i) for i in range(int(start,16), int(stop,16))])

PUNCTUATION_START = re.compile(ur'^([{}\u2000-\u206F\u3000-\u303F\uFF00-\uFFFF]*)'.format(string.punctuation))
PUNCTUATION_END = re.compile(ur'([{}\u2000-\u206F\u3000-\u303F\uFF00-\uFFFF]*)$'.format(string.punctuation))
PUNC = unicode(string.punctuation) + _unicode_range('2000','206F') + _unicode_range('3000', '303F') + _unicode_range('FF00', 'FFFF')
PUNC_TABLE = {ord(c): None for c in PUNC}

def strip_punc(tsent):
    """
    """
    out = []
    for word in tsent:
        w = strip_punc_word(word)
        if w:
            out.append(w)

    return out

def strip_punc_word(word):
    if isinstance(word, unicode):
        return word.translate(PUNC_TABLE)
    elif isinstance(word, str):
        return word.translate(None, string.punctuation)

NUMS = string.digits
NUMS_TABLE =  {ord(c): None for c in NUMS}

def rem_num(tsent):
    """
    """
    #p = re.compile(r'(^\D+$)|(^\D*[1-2]\d\D*$|^\D*\d\D*$)')
    out = []
    for word in tsent:
        w = rem_num_word(word)
        if w:
            out.append(w)

    return out

def rem_num_word(word):
    if isinstance(word, unicode):
        return word.translate(NUMS_TABLE)
    elif isinstance(word, str):
        return word.translate(None, string.digits)

def rehyph(sent):
    """
    """
    return re.sub(r'(?P<x1>.)--(?P<x2>.)', '\g<x1> - \g<x2>', sent)


BIG_TABLE = NUMS_TABLE.copy()
BIG_TABLE.update(PUNC_TABLE)
BIG_LIST = string.digits + string.punctuation
def process_word(word):
    if isinstance(word, unicode):
        return word.translate(BIG_TABLE)
    elif isinstance(word, str):
        return word.translate(None, BIG_LIST)

def apply_stoplist(corp, nltk_stop=True, add_stop=None, freq=0, in_place=True):
    """
    Returns a Corpus object with stop words eliminated.

    :param corp: Corpus object to apply stoplist to.
    :type corp: Corpus

    :param nltk_stop: If `True` English stopwords from nltk are included
        in the stoplist. Default is `True`.
    :type nltk_stop: boolean, optional
    
    :param add_stop: list of words to eliminate from `corp` words.
        Default is `None`.
    :type add_stop: List, optional

    :param freq: Eliminates words that appear <= `freq` times. Default is
        0.
    :type freq: int

    :returns: Corpus with words in the stoplist removed.

    :See Also: :class:`vsm.corpus.Corpus`, :meth:`vsm.corpus.Corpus.apply_stoplist`
    """
    stoplist = set()
    if nltk_stop:
        import nltk
        for w in nltk.corpus.stopwords.words('english'):
            stoplist.add(w)
    if add_stop:
        for w in add_stop:
            stoplist.add(w)

    if not in_place:
        return corp.apply_stoplist(stoplist=stoplist, freq=freq)
    else:
        corp.in_place_stoplist(stoplist=stoplist, freq=freq)
        return corp

def in_place_stoplist(corp, nltk_stop=True, add_stop=None, freq=0):
    """
    Returns a Corpus object with stop words eliminated.

    :param corp: Corpus object to apply stoplist to.
    :type corp: Corpus

    :param nltk_stop: If `True` English stopwords from nltk are included
        in the stoplist. Default is `True`.
    :type nltk_stop: boolean, optional
    
    :param add_stop: list of words to eliminate from `corp` words.
        Default is `None`.
    :type add_stop: List, optional

    :param freq: Eliminates words that appear <= `freq` times. Default is
        0.
    :type freq: int

    :returns: Corpus with words in the stoplist removed.

    :See Also: :class:`vsm.corpus.Corpus`, :meth:`vsm.corpus.Corpus.apply_stoplist`
    """
    stoplist = set()
    if nltk_stop:
        import nltk
        for w in nltk.corpus.stopwords.words('english'):
            stoplist.add(w)
    if add_stop:
        for w in add_stop:
            stoplist.add(w)

    corp.in_place_stoplist(stoplist=stoplist, freq=freq)


def filter_by_suffix(l, ignore, filter_dotfiles=True):
    """
    Returns elements in `l` that does not end with elements in `ignore`,
    and filters dotfiles (files that start with '.').

    :param l: List of strings to filter.
    :type l: list

    :param ignore: List of suffix to be ignored or filtered out.
    :type ignore: list

    :param filter_dotfiles: Filter dotfiles.
    :type filter_dotfiles: boolean, default True

    :returns: List of elements in `l` whose suffix is not in `ignore`.

    **Examples**

    >>> l = ['a.txt', 'b.json', 'c.txt']
    >>> ignore = ['.txt']
    >>> filter_by_suffix(l, ignore)
    ['b.json']
    """
    filter_list = [e for e in l if not sum([e.endswith(s) for s in ignore])]
    if filter_dotfiles:
        filter_list = [e for e in filter_list if not e.startswith('.')]
    return filter_list

class _tokenizer:
    def tokenize(self, text):
        return text.split()

word_tokenizer = _tokenizer()
def word_tokenize(text):
    """Takes a string and returns a list of strings. Intended use: the
    input string is English text and the output consists of the
    lower-case words in this text with numbers and punctuation, except
    for hyphens, removed.

    The core work is done by NLTK's Treebank Word Tokenizer.
    
    :param text: Text to be tokeized.
    :type text: string

    :returns: tokens : list of strings
    """
    global word_tokenizer
    if word_tokenizer is None:
        import nltk
        word_tokenizer = nltk.TreebankWordTokenizer()

    text = rehyph(text)
    text = process_word(text)
    text = text.replace(u'\x00','')
    text = text.lower()
    tokens = word_tokenizer.tokenize(text)

    #process_word = lambda x: strip_punc_word(rem_num_word(word)).lower().replace(u'\x00','')
    #tokens = [process_word(word) for word in text]

    return tokens


sent_tokenizer = None
def sentence_tokenize(text):
    """
    Takes a string and returns a list of strings. Intended use: the
    input string is English text and the output consists of the
    sentences in this text.

    This is a wrapper for NLTK's pre-trained Punkt Tokenizer.
     
    :param text: Text to be tokeized.
    :type text: string

    :returns: tokens : list of strings
    """
    global sent_tokenizer
    if sent_tokenizer is None:
        import nltk
        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    return sent_tokenizer.tokenize(text)

def sentence_span_tokenize(text):
    """
    Takes a string and returns a list of strings. Intended use: the
    input string is English text and the output consists of the
    sentences in this text.

    This is a wrapper for NLTK's pre-trained Punkt Tokenizer.
     
    :param text: Text to be tokeized.
    :type text: string

    :returns: token_spans : iterator of (start,stop) tuples.
    """
    global sent_tokenizer
    if sent_tokenizer is None:
        import nltk
        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    return sent_tokenizer.span_tokenize(text)



def paragraph_tokenize(text):
    """
    Takes a string and returns a list of strings. Intended use: the
    input string is English text and the output consists of the
    paragraphs in this text. It's expected that the text marks
    paragraphs with two consecutive line breaks.
     
    :param text: Text to be tokeized.
    :type text: string

    :returns: tokens : list of strings
    """

    par_break = re.compile(r'[\r\n]{2,}')
    span_start = 0
    for match in par_break.finditer(text):
        span_end = match.span()[0]
        yield text[span_start:span_end]
        span_start = match.span()[1]
    yield text[span_start:]
    #return par_break.split(text)


def detect_encoding(filename):
    """
    Takes a filename and attempts to detect the character encoding of the file
    using `chardet`.
     
    :param filename: Name of the file to process
    :type filename: string

    :returns: encoding : string
    """
    detector = UniversalDetector()
    with open(filename, 'rb') as unknown_file:
        for line in unknown_file:
            detector.feed(line)
            if detector.done:
                break
    detector.close()

    return detector.result['encoding']
     
