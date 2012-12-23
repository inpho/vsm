#!/usr/bin/python

import os

import argparse

from vsm.corpus import Corpus
from vsm.model.ldagibbs import LDAGibbs as LDA
from vsm.util.batchtools import LDA_trainer



def job(K, batches, batch_size):

    root = '/home/rrose1/develop/vsm-tests/'

    corpus_file = os.path.join(root, 'hathi-test.npz')

    prefix = 'hathi-test-LDA-K' + str(K)

    c = Corpus.load(corpus_file)

    m = LDA(c, tok_name='book', K=K)

    LDA_trainer(m, root, prefix, batches=batches, batch_size=batch_size)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('K', type=int, default=100)

    parser.add_argument('-n', '--n_batches', type=int, default=2)

    parser.add_argument('-s', '--batch_size', type=int, default=2)

    args = parser.parse_args()

    job(args.K, args.n_batches, args.batch_size)

