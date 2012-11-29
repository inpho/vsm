import os
import re

import numpy as np

import nltk



#
# Tokenizing functions
#



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



def textfile_tokenize(path):
    """
    Takes a string and returns a list of strings and a dictionary.
    Intended use: the input string is a directory name containing
    plain text files. The output list is a list of strings, each of
    which is the contents of one of these files. The dictionary is a
    map from indices of this list to the names of the source files.
    """

    out = [],{}
    
    filenames = os.listdir(path)

    for filename in filenames:
        
        filename = os.path.join(path, filename)

        with open(filename, mode='r') as f:

            out[0].append(f.read())

            out[1][len(out[0]) - 1] = filename

    return out



#
# Tokenizing classes
#



class MultipleArticleTokenizer(object):
    """
    """
    def __init__(self, path):

        self.path = path

        self.words = []

        self.tok_names = ['articles', 'paragraphs', 'sentences']

        self.tok_data = None

        self._compute_tokens()
    


    def _compute_tokens(self):

        articles, articles_metadata = textfile_tokenize(self.path)

        article_tokens = []

        paragraph_tokens = []

        sentence_spans = []

        print 'Computing article and paragraph tokens'

        for i,article in enumerate(articles):

            print 'Processing article in', articles_metadata[i]

            paragraphs = paragraph_tokenize(article)
            
            for paragraph in paragraphs:
                
                sentences = sentence_tokenize(paragraph)

                for sentence in sentences:
                    
                    words = word_tokenize(sentence)

                    self.words.extend(words)
                    
                    sentence_spans.append(len(words))

                paragraph_tokens.append(sum(sentence_spans))
                    
            article_tokens.append(sum(sentence_spans))

        print 'Computing sentence tokens'

        sentence_tokens = np.cumsum(sentence_spans)

        article_tokens = zip(article_tokens, articles_metadata)

        self.tok_data = [article_tokens, paragraph_tokens, sentence_tokens]



class SingleArticleTokenizer(object):
    """
    """
    def __init__(self, filename):

        self.filename = filename

        self.words = []

        self.tok_names = ['paragraphs', 'sentences']

        self.tok_data = None

        self._compute_tokens()

        

    def _compute_tokens(self):

        with open(self.filename, mode='r') as f:

            article = f.read()

        paragraph_tokens = []

        sentence_spans = []

        print 'Computing paragraph tokens'

        paragraphs = paragraph_tokenize(article)
            
        for paragraph in paragraphs:

            sentences = sentence_tokenize(paragraph)

            for sentence in sentences:

                words = word_tokenize(sentence)

                self.words.extend(words)
                    
                sentence_spans.append(len(words))

            paragraph_tokens.append(sum(sentence_spans))
                    
        print 'Computing sentence tokens'

        sentence_tokens = np.cumsum(sentence_spans)

        self.tok_data = [paragraph_tokens, sentence_tokens]



def toy_corpus(plain_corpus, is_filename=False, 
               nltk_stop=False, mask_freq=0, add_stop=None):
    """
    Takes plain text corpus as a string. Document tokens are delimited
    by `\n\n`. E.g.,

    <document 0>

    <document 1>

    ...

    <document n>

    where <document i> is any chunk of text to be tokenized by word.

    If `is_filename` is True then `plain_corpus` is intended to be a filename.
    """
    from vsm.corpus import MaskedCorpus, mask_from_stoplist, mask_freq_t

    if is_filename:

        with open(plain_corpus, 'r') as f:

            plain_corpus = f.read()

    docs = plain_corpus.split('\n\n')

    docs = [word_tokenize(d) for d in docs]

    corpus = sum(docs, [])

    tok = np.cumsum(np.array([len(d) for d in docs]))

    tok = [(i, str(d)) for d, i in enumerate(tok)]
    
    c = MaskedCorpus(corpus, tok_data=[tok], tok_names=['documents'])

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

    c = c.to_corpus(compress=True)

    return c
    


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

    assert toy_corpus(keats, mask_freq=1)

    assert toy_corpus(keats, add_stop=['and', 'with'])

    assert toy_corpus(keats, nltk_stop=True,
                      mask_freq=1, add_stop=['ay'])

    from tempfile import NamedTemporaryFile as NFT

    tmp = NFT(delete=False)

    tmp.write(keats)

    tmp.close()

    c = toy_corpus(tmp.name, is_filename=True, 
                   nltk_stop=True, add_stop=['ay'])
    
    assert c

    os.remove(tmp.name)

    return c
