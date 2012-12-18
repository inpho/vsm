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



def txt_tokenize(chunks, annotations, chunk_name='page', paragraphs=True):

    words, chk_tokens, sent_tokens = [], [], []

    sent_break, chk_n, sent_n = 0, 0, 0

    if paragraphs:

        par_tokens = []

        par_n = 0
        
        for chk, annot in zip(chunks, annotations):

            pars = paragraph_tokenize(chk)

            for par in pars:
                
                sents = sentence_tokenize(par)

                for sent in sents:
                    
                    w = word_tokenize(sent)
                    
                    words.extend(w)
                    
                    sent_break += len(w)
                    
                    sent_tokens.append((sent_break, annot, 
                                        chk_n, par_n, sent_n))

                    sent_n += 1

                par_tokens.append((sent_break, annot, chk_n, par_n))

                par_n += 1

            chk_tokens.append((sent_break, annot, chk_n))

            chk_n += 1

    else:

        for chk, annot in zip(chunks, annotations):

            sents = sentence_tokenize(chk)

            for sent in sents:
                    
                w = word_tokenize(sent)
                    
                words.extend(w)
                    
                sent_break += len(w)

                sent_tokens.append((sent_break, annot, chk_n, sent_n))

                sent_n += 1

            chk_tokens.append((sent_break, annot, chk_n))

            chk_n += 1

    idx_dt = ('idx', np.int32)

    annot_dt = (chunk_name + '_annot', np.array(annotations).dtype)

    chk_n_dt = (chunk_name + '_n', np.int32)

    sent_n_dt = ('sent_n', np.int32)

    corpus_data = dict()

    dtype = [idx_dt, annot_dt, chk_n_dt]

    corpus_data[chunk_name] = np.array(chk_tokens, dtype=dtype)

    if paragraphs:

        par_n_dt = ('par_n', np.int32)

        dtype = [idx_dt, annot_dt, chk_n_dt, par_n_dt]

        corpus_data['paragraph'] = np.array(par_tokens, dtype=dtype)

        dtype = [idx_dt, annot_dt, chk_n_dt, par_n_dt, sent_n_dt]

        corpus_data['sentence'] = np.array(sent_tokens, dtype=dtype)

    else:

        dtype = [idx_dt, annot_dt, chk_n_dt, sent_n_dt]

        corpus_data['sentence'] = np.array(sent_tokens, dtype=dtype)

    return words, corpus_data



def test_txt_tokenize():

    chunks = ['foo foo foo\n\nfoo foo',
             'Foo bar.  Foo bar.', 
             '',
             'foo\n\nfoo']

    annotations = [str(i) for i in xrange(len(chunks))]

    words, tok_data = txt_tokenize(chunks, annotations)

    assert len(words) == 11

    assert len(tok_data['page']) == 4

    assert len(tok_data['paragraph']) == 6

    assert len(tok_data['sentence']) == 7
    
    assert (tok_data['page']['idx'] == [5, 9, 9, 11]).all()

    assert (tok_data['page']['page_annot'] == ['0', '1', '2', '3']).all()

    assert (tok_data['page']['page_n'] == [0, 1, 2, 3]).all()

    assert (tok_data['paragraph']['idx'] == [3, 5, 9, 9, 10, 11]).all()

    assert (tok_data['paragraph']['page_annot'] == 
            ['0', '0', '1', '2', '3', '3']).all()

    assert (tok_data['paragraph']['page_n'] == 
            [0, 0, 1, 2, 3, 3]).all()

    assert (tok_data['paragraph']['par_n'] == 
            [0, 1, 2, 3, 4, 5]).all()

    assert (tok_data['sentence']['idx'] == [3, 5, 7, 9, 9, 10, 11]).all()

    assert (tok_data['sentence']['page_annot'] == 
            ['0', '0', '1', '1', '2', '3', '3']).all()

    assert (tok_data['sentence']['page_n'] == 
            [0, 0, 1, 1, 2, 3, 3]).all()

    assert (tok_data['sentence']['par_n'] == 
            [0, 1, 2, 2, 3, 4, 5]).all()

    assert (tok_data['sentence']['sent_n'] == 
            [0, 1, 2, 3, 4, 5, 6]).all()




def txt_corpus(plain_dir, chunk_name='articles', paragraphs=True,
               compress=True, nltk_stop=True, mask_freq=1, add_stop=None):
    """
    `txt_corpus` is a convenience function for generating Corpus or
    MaskedCorpus objects from a directory of plain text files.

    `txt_corpus` will retain file-level tokenization and perform
    sentence and word tokenizations. Optionally, it will provide
    paragraph-level tokenizations.

    It will also strip punctuation and arabic numerals outside the
    range 1-29. All letters are made lowercase.

    Parameters
    ----------
    plain_dir : string-like
        String containing directory containing a plain-text corpus.
    chunk_name : string-line
        The name of the tokenization corresponding to individual
        files. For example, if the files are pages of a book, one
        might set `chunk_name` to `pages`. Default is `articles`.
    paragraphs : boolean
        If `True`, a paragraph-level tokenization is included.
        Defaults to `True`.
    compress : boolean
        If `True` then a Corpus object is returned with all masked
        terms removed. Otherwise, a MaskedCorpus object is returned.
        Default is `True`.
    nltk_stop : boolean
        If `True` then the corpus object is masked using the NLTK
        English stop words. Default is `False`.
    mask_freq : int
        The upper bound for a term to be masked on the basis of its
        collection frequency. Default is 0.
    add_stop : array-like
        A list of stop words. Default is `None`.

    Returns
    -------
    c : a Corpus or a MaskedCorpus object
        Contains the tokenized corpus built from the input plain-text
        corpus. Document tokens are named `documents`.

    """

    from vsm.corpus import MaskedCorpus

    chunks = []
    
    filenames = os.listdir(plain_dir)

    filenames.sort()

    for filename in filenames:
        
        filename = os.path.join(plain_dir, filename)

        with open(filename, mode='r') as f:

            chunks.append(f.read())

    words, tok = txt_tokenize(chunks, filenames, chunk_name=chunk_name,
                              paragraphs=paragraphs)

    names, data = zip(*tok.items())
    
    c = MaskedCorpus(words, tok_data=data, tok_names=names)

    c = mask_corpus(c, nltk_stop=nltk_stop,
                    mask_freq=mask_freq, add_stop=add_stop)

    if compress:

        c = c.to_corpus(compress=True)

    return c



def toy_corpus(plain_corpus, is_filename=False, compress=True,
               nltk_stop=False, mask_freq=0, add_stop=None, metadata=None):
    """
    `toy_corpus` is a convenience function for generating Corpus or
    MaskedCorpus objects from a given string or a single file.

    `toy_corpus` will perform both word and document-level
    tokenization. It will also strip punctuation and arabic numerals
    outside the range 1-29. All letters are made lowercase.

    Document tokens are delimited by `\n\n`. E.g.,

        <document 0>

        <document 1>

        ...

        <document n>

    where <document i> is any chunk of text to be tokenized by word.

    Parameters
    ----------
    plain_corpus : string-like
        String containing a plain-text corpus or a filename of a file
        containing one.
    is_filename : boolean
        If `True` then `plain_corpus` is treated like a filename.
        Otherwise, `plain_corpus` is presumed to contain the corpus.
        Default is `False`.
    compress : boolean
        If `True` then a Corpus object is returned with all masked
        terms removed. Otherwise, a MaskedCorpus object is returned.
        Default is `True`.
    nltk_stop : boolean
        If `True` then the corpus object is masked using the NLTK
        English stop words. Default is `False`.
    mask_freq : int
        The upper bound for a term to be masked on the basis of its
        collection frequency. Default is 0.
    add_stop : array-like
        A list of stop words. Default is `None`.
    metadata : array-like
        A list of strings providing metadata about the documents. If
        provided, must have length equal to the number of documents.
        Default is `None`.
    Returns
    -------
    c : a Corpus or a MaskedCorpus object
        Contains the tokenized corpus built from the input plain-text
        corpus. Document tokens are named `documents`.

    """
    from vsm.corpus import MaskedCorpus, mask_from_stoplist, mask_freq_t

    if is_filename:

        with open(plain_corpus, 'r') as f:

            plain_corpus = f.read()

    docs = plain_corpus.split('\n\n')

    docs = [word_tokenize(d) for d in docs]

    corpus = sum(docs, [])

    tok = np.cumsum(np.array([len(d) for d in docs]))

    if metadata:

        if not len(metadata) == len(tok):

            msg = 'Metadata mismatch: metadata length is {0} and number'\
                   'of documents is {1}'.format(len(metadata), len(tok))

            raise Exception(msg)

        else:

            dtype = [('idx', np.array(tok).dtype),
                     ('short_label', np.array(metadata).dtype)]

            tok = np.array(zip(tok, metadata), dtype=dtype)

    else:

        dtype = [('idx', np.array(tok).dtype)]

        tok = np.array([(i,) for i in tok], dtype=dtype)
    
    c = MaskedCorpus(corpus, tok_data=[tok], tok_names=['documents'])

    c = mask_corpus(c, nltk_stop=nltk_stop,
                    mask_freq=mask_freq, add_stop=add_stop)

    if compress:

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



#
# Tokenizing classes
#
# (These will be superseded by functions such as those above.)


# class MultipleArticleTokenizer(object):
#     """
#     """
#     def __init__(self, path):

#         self.path = path

#         self.words = []

#         self.tok_names = ['articles', 'paragraphs', 'sentences']

#         self.tok_data = None

#         self._compute_tokens()
    


#     def _compute_tokens(self):

#         articles, articles_metadata = textfile_tokenize(self.path)

#         article_tokens = []

#         paragraph_tokens = []

#         sentence_spans = []

#         print 'Computing article and paragraph tokens'

#         for i,article in enumerate(articles):

#             print 'Processing article in', articles_metadata[i]

#             paragraphs = paragraph_tokenize(article)
            
#             for paragraph in paragraphs:
                
#                 sentences = sentence_tokenize(paragraph)

#                 for sentence in sentences:
                    
#                     words = word_tokenize(sentence)

#                     self.words.extend(words)
                    
#                     sentence_spans.append(len(words))

#                 paragraph_tokens.append(sum(sentence_spans))
                    
#             article_tokens.append(sum(sentence_spans))

#         print 'Computing sentence tokens'

#         sentence_tokens = np.cumsum(sentence_spans)

#         article_tokens = zip(article_tokens, articles_metadata)

#         self.tok_data = [article_tokens, paragraph_tokens, sentence_tokens]



# class SingleArticleTokenizer(object):
#     """
#     """
#     def __init__(self, filename):

#         self.filename = filename

#         self.words = []

#         self.tok_names = ['paragraphs', 'sentences']

#         self.tok_data = None

#         self._compute_tokens()

        

#     def _compute_tokens(self):

#         with open(self.filename, mode='r') as f:

#             article = f.read()

#         paragraph_tokens = []

#         sentence_spans = []

#         print 'Computing paragraph tokens'

#         paragraphs = paragraph_tokenize(article)
            
#         for paragraph in paragraphs:

#             sentences = sentence_tokenize(paragraph)

#             for sentence in sentences:

#                 words = word_tokenize(sentence)

#                 self.words.extend(words)
                    
#                 sentence_spans.append(len(words))

#             paragraph_tokens.append(sum(sentence_spans))
                    
#         print 'Computing sentence tokens'

#         sentence_tokens = np.cumsum(sentence_spans)

#         self.tok_data = [paragraph_tokens, sentence_tokens]
