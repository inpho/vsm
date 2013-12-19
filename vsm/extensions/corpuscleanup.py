import nltk
from vsm.corpus import Corpus
from vsm.corpus.util import *


def apply_stoplist_len(corp, nltk_stop=True, add_stop=None, 
                    word_len=3, freq=0):
    """
    New parameter word_len is added. Adds words with length
    <= word_len to the stoplist. A rough solution for
    getting rid of bibliographic information and common
    foreign language particles.
    """ 
    stoplist = set()
    if nltk_stop:
        for w in nltk.corpus.stopwords.words('english'):
            stoplist.add(w)
    if add_stop:
        for w in add_stop:
            stoplist.add(w)
    for w in corp.words: 
        if len(w) <= word_len:
            stoplist.add(w)

    return corp.apply_stoplist(stoplist=stoplist, freq=freq)


def apply_stoplist_nltk(corp, nltk_stop='english', add_stop=None, 
                    word_len=3, freq=0):
    """
    Originally nltk_stop was a boolean that filtered 'english'.
    Now it is a string, language, supported in nltk.corpus.
    stopwords. If nltk_stop is set to None, then no stopwords
    will be added from nltk corpus.
    """ 
    stoplist = set()
    if nltk_stop:
        try:
            for w in nltk.corpus.stopwords.words('english'):
                stoplist.add(w)
        except Exception:
            print "{0} language not found in nltk.corpus\
                    .stopwords".format(nltk_stop)
    if add_stop:
        for w in add_stop:
            stoplist.add(w)
    for w in corp.words: 
        if len(w) <= word_len:
            stoplist.add(w)

    return corp.apply_stoplist(stoplist=stoplist, freq=freq)



