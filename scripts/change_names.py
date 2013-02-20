#!usr/bin/python 

import numpy as np
import argparse

def change_names(fname):
    """
    Takes a file name and updates the old terms
    """
    l = np.load(fname)
    
    words = l['terms']
    cd_sent = l['tok_data_sentence']
    cd_art = l['tok_data_article']
    cd_para = l['tok_data_paragraph']
    ctx_types = l['tok_names']
    corpus = l['corpus'] 

    np.savez(fname, words=words, context_data_sentence=cd_sent,
	context_data_article=cd_art, context_data_paragraph=cd_para,
	context_types=ctx_types, corpus=corpus)

if __name__=="__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('fname')
    args = parser.parse_args()

    change_names(args.fname)
