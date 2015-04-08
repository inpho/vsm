"""
vsm.extensions.interop.ldac

Module containing functions for import/export between VSM and lda-c, which is
the original LDA implementation referenced in Blei, Ng, and Jordan (2003). 
lda-c is available at: http://www.cs.princeton.edu/~blei/lda-c/
"""
import os, os.path

from scipy.stats import itemfreq
import numpy as np

def export_corpus(corpus, outfolder, context_type='document'):
    """
    Converts a vsm.corpus.Corpus object into a lda-c compatible data file.
    Creates two files:
    1.  "vocab.txt" - contains the integer-word mappings
    2.  "corpus.dat" - contains the corpus object in the format described in 
        [lda-c documentation](http://www.cs.princeton.edu/~blei/lda-c/readme.txt):

            Under LDA, the words of each document are assumed exchangeable.  Thus,
            each document is succinctly represented as a sparse vector of word
            counts. The data is a file where each line is of the form:
        
                [M] [term_1]:[count] [term_2]:[count] ...  [term_N]:[count]
        
            where [M] is the number of unique terms in the document, and the
            [count] associated with each term is how many times that term appeared
            in the document.  Note that [term_1] is an integer which indexes the
            term; it is not a string.

    :param corpus: VSM Corpus object to convert to lda-c file
    :type corpus: vsm.corpus.Corpus

    :param outfolder: Directory to output "vocab.txt" and "corpus.dat"
    :type string: path
    """
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    vocabfilename = os.path.join(outfolder, 'vocab.txt')
    with open(vocabfilename,'w') as vocabfile:
        for word in corpus.words:
            vocabfile.write(word+'\n')

    corpusfilename = os.path.join(outfolder, 'corpus.dat')
    with open(corpusfilename,'w') as corpusfile:
        for ctx in corpus.view_contexts(context_type):
            M = len(np.unique(ctx))
            corpusfile.write("{0}".format(M))

            for token in itemfreq(ctx):
                corpusfile.write(" {term}:{count}".format(
                    term=token[0],count=token[1]))

            corpusfile.write("\n")
            

def import_corpus(vocabfilename, corpusfilename):
    pass

def import_model(filename):
    pass
def export_model(filename):
    pass
