import os
import re

import numpy as np

import nltk



def strip_punc(tsent):

    p1 = re.compile(r'^(\W*)')

    p2 = re.compile(r'(\W*)$')

    out = []

    for word in tsent:

        w = re.sub(p2, '', re.sub(p1, '', word))

        if w:

            out.append(w)

    return out



def rem_num(tsent):

    p = re.compile(r'(^\D+$)|(^\D*[1-2]\d\D*$|^\D*\d\D*$)')

    return [word for word in tsent if re.search(p, word)]



def rehyph(sent):

    return re.sub(r'(?P<x1>.)--(?P<x2>.)', '\g<x1> - \g<x2>', sent)



def word_tokenize(text):
    """
    Takes a string and returns a list of strings. Intended use: the
    input string is English text and the output consists of the
    lower-case words in this text with numbers and punctuation, except
    for hyphens, removed.

    The core work is done by NLTK's Treebank Word Tokenizer.
    """

    text = rehyph(text)

    text = nltk.TreebankWordTokenizer().tokenize(text)

    tokens = [word.lower() for word in text]

    tokens = strip_punc(tokens)

    tokens = rem_num(tokens)
    
    return tokens



def sentence_tokenize(text):
    """
    Takes a string and returns a list of strings. Intended use: the
    input string is English text and the output consists of the
    sentences in this text.

    This is a wrapper for NLTK's pre-trained Punkt Tokenizer.
    """
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    return tokenizer.tokenize(text)



def paragraph_tokenize(text):
    """
    Takes a string and returns a list of strings. Intended use: the
    input string is English text and the output consists of the
    paragraphs in this text. It's expected that the text marks
    paragraphs with two consecutive line breaks.
    """
    
    return text.split('\n\n')



def textfile_tokenize(path, sort=False):
    """
    Takes a string and returns a list of strings and a dictionary.
    Intended use: the input string is a directory name containing
    plain text files. The output list is a list of strings, each of
    which is the contents of one of these files. The dictionary is a
    map from indices of this list to the names of the source files.
    """

    out = [],{}
    
    filenames = os.listdir(path)

    if sort:

        filenames.sort()

    for filename in filenames:
        
        filename = os.path.join(path, filename)

        with open(filename, mode='r') as f:

            out[0].append(f.read())

            out[1][len(out[0]) - 1] = filename

    return out




def mask_corpus(c, nltk_stop=False, mask_freq=0, add_stop=None):

    from vsm.corpus import mask_from_stoplist, mask_freq_t

    stoplist = set()

    if nltk_stop:

        for w in nltk.corpus.stopwords.words('english'):

            stoplist.add(w)

    if add_stop:

        for w in add_stop:

            stoplist.add(w)

    if stoplist:

        mask_from_stoplist(c, list(stoplist))

    if mask_freq > 0:

        mask_freq_t(c, mask_freq)

    return c



def filter_by_suffix(l, ignore):

    return [e for e in l if not sum([e.endswith(s) for s in ignore])]



