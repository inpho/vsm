#!/usr/bin/python 

import numpy as np
import argparse

def change_names(fname):
    """
    Takes a file name and updates the old terms
    """
    l = np.load(fname)

    store = [(k, l[k]) for k in l.files]
    i = store.index(('tok_name', l['tok_name']))
    store[i] = ('context_type', l['tok_name'])
    
    np.savez(fname, **dict(store))

if __name__=="__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('fname')
    args = parser.parse_args()

    change_names(args.fname)
