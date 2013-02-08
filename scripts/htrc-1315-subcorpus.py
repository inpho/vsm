import os
import numpy as np

root = '/var/inphosemantics/data/20130101/'
subcorpus = 'htrc-anthropomorphism-86/plain/'
main_corpus = 'htrc-anthropomorphism-1315/plain/'
vol_list = 'htrc-anthropomorphism-86/htrc-subset-k=8_10_16-alpha=.25.npy'

l = np.load(root + vol_list)

for vol in l:
    os.symlink(root + main_corpus + vol, root + subcorpus + vol)
