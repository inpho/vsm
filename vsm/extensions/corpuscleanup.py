import nltk
from vsm.corpus import Corpus
from vsm.extensions.corpusbuilders.util import *



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


def apply_stoplist_nltk(corp, nltk_stop=[], add_stop=None, 
                    word_len=0, freq=0):
    """
    Originally nltk_stop was a boolean that filtered 'english'.
    Now it is a string, language, supported in nltk.corpus.
    stopwords. If nltk_stop is set to None, then no stopwords
    will be added from nltk corpus.
    """ 
    stoplist = set()
    if len(nltk_stop) > 0:
        for lang in nltk_stop:
            try:
                for w in nltk.corpus.stopwords.words(lang):
                    stoplist.add(w)
            except Exception:
                print "{0} language not found in nltk.corpus\
                        .stopwords".format(nltk_stop)
    if add_stop:
        for w in add_stop:
            stoplist.add(w)
    
    if word_len > 0:
        for w in corp.words: 
            if len(w) <= word_len:
                stoplist.add(w)

    return corp.apply_stoplist(stoplist=stoplist, freq=freq)


def snowball_stem(corp, language='english'):
    """
    Builds a dictionary with words as keys and stems as the values.
    language : string. 'english', 'german', or 'french'.
    """
    stemmer = []
    if language == 'english':
        stemmer = nltk.stem.snowball.EnglishStemmer()
    elif language == 'german':
        stemmer = nltk.stem.snowball.GermanStemmer()
    elif language == 'french':
        stemmer = nltk.stem.snowball.FrenchStemmer()

    stemdict = {}
    for w in corp.words:
        w_ = w.decode('utf-8').strip()
        stemdict[w] = unidecode(stemmer.stem(w_))

    return stemdict


def porter_stem(corp):
    """
    Builds a dictionary with words as keys and stems as the values.
    """
    from porterstemmer import PorterStemmer

    ps = PorterStemmer()
    psdict = {}
    for w in corp.words:
        psdict[w] = ps.stem(w)
    
    return psdict


def stem_int(corp, stemdict):
    """
    Returns a dictionary to replace corp.words_int
    """

    wordint = {}
    sint = -1
    prev = ''
    for k in corp.words:
        stem = stemdict[k]
        
        if k == stem:
            wordint[k] = corp.words_int[k]
            sint = -1
            
        else: # replace it with stem
            if stem in corp.words:
                wordint[k] = corp.words_int[stem]
                sint = -1
                
            else: # create a new entry or new int
                if sint in wordint.values() and prev == stem: 
                    wordint[k] = sint

                else: # new stem, new sint
                    wordint[k] = corp.words_int[k]
                    sint = wordint[k]

        prev = stem
 
    return wordint


def word_stem(corp, stemdict):
    """
    Returns a dictionary with integer maps in corp.words_int
    as keys and integers for stems as values.
    """

    intint = {}
    sint = -1
    prev = ''
    for k in corp.words:
        stem = stemdict[k]
        orig = corp.words_int[k]
        
        if k == stem: # same as c.words_int
            intint[orig] = corp.words_int[k]
            sint = -1
            
        else: # replace it with stem
            if stem in corp.words:
                intint[orig] = corp.words_int[stem]
                sint = -1
                
            else: # create a new entry or new int            
                if sint in intint.values() and prev == stem:
                    intint[orig] = sint
                else:
                    intint[orig] = corp.words_int[k]
                    sint = intint[orig]
        prev = stem
        
    return intint
