import numpy as np

import vsm

from vsm import model
from vsm.model import beagleenvironment as be
from vsm.model import beaglecontext as bc
from vsm.model import beagleorder as bo



class BeagleComposite(model.Model):

    def train(self,
              corpus,
              tok_name='sentences',
              n_columns=2048,
              env_matrix=None,
              ctx_matrix=None,
              ord_matrix=None,
              lmda=7,
              psi=None,
              rand_perm=None):


        if ctx_matrix is None or ord_matrix is None:

            if env_matrix is None:

                print 'Generating environment vectors'

                env_model = be.BeagleEnvironment()

                env_model.train(corpus, n_columns)

                env_matrix = env_model.matrix[:, :]

            print 'Computing context vectors'
                
            ctx_model = bc.BeagleContext()

            ctx_model.train(corpus,
                            tok_name=tok_name,
                            env_matrix=env_matrix)

            ctx_matrix = ctx_model.matrix[:, :]

            print 'Computing order vectors'

            ord_model = bo.BeagleOrder()

            ord_model.train(corpus,
                            tok_name=tok_name,
                            env_matrix=env_matrix,
                            lmda=lmda,
                            psi=psi,
                            rand_perm=rand_perm)

            ord_matrix = ord_model.matrix[:, :]

        print 'Normalizing and summing context and order vectors'

        ctx_matrix = vsm.row_normalize(ctx_matrix)

        ord_matrix = vsm.row_normalize(ord_matrix)
        
        self.matrix = ctx_matrix

        self.matrix += ord_matrix



#
# for testing
#

def test_BeagleComposite():

    from vsm import corpus

    n = 256

    c = corpus.random_corpus(1e5, 1e2, 1, 10, tok_name='sentences')

    m = BeagleComposite()

    m.train(c, n_columns=n)

    return m
