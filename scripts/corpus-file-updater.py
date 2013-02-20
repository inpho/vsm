#!/usr/bin/python 

import numpy as np
import argparse

def change_names(fname):
    """
    Takes a file name and updates the old terms
    """
    l = np.load(fname)
    
    words = l['terms']
    ctx_types = l['tok_names']
    corpus = l['corpus'] 

    store = {}
    for t in ctx_types:
	name = l['tok_data_' + t]
	store['context_data_' + t] = name

    np.savez(fname, words=words, context_types=ctx_types, 
		corpus=corpus, **store)

if __name__=="__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('fname')
    args = parser.parse_args()

    change_names(args.fname)
