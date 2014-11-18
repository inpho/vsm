from FileReadWrite import *
from py4j.java_gateway import JavaGateway
import numpy as np


def lda_run(corp, corpfile, ctx_type, iterations, K, metafile,
            alpha=0.01, beta=0.01):
    """
    corp : `vsm.Corpus`
    corpfile : fname, '\n' indicates ctx breaks (tokenizaiton).
    ctx_type : a context type in corp.context_types
    
    """
    
    ## This is redundant since java program is running with an existing corp.
    ## write_file(corp, ctx_type, corpfile)

    # connect to java
    gw = JavaGateway(auto_field=True)

    # gibbs sampling
    gw.entry_point.sample(iterations, K, alpha, beta)
    # write necessary data for vsm.model.ldagibbs (for saving step)
    gw.entry_point.writeMeta(iterations, K, alpha, beta, metafile)
    
    # LdaGibbsSampler object 
    lda = gw.entry_point.getLda()

    return gw, lda


def save(lda, ctx_type, fname, metafile):
    """
    lda : LdaGibbsSampler obj
    fname : output file
    metafile : contains log_probs, top_doc,,and such 
    """
    top_word = nested_arr_to_np(lda.getPhi())
    doc_top = nested_arr_to_np(lda.getTheta())
    W = nested_arr_to_np(lda.getDocuments(), arrarr=True)
    Z = nested_arr_to_np(lda.getZ(), arrarr=True)
    dic = file_to_dict(metafile)
    
    arrays_out = dict()
    arrays_out['W_corpus'] = np.array(np.hstack(W))
    arrays_out['W_indices'] = np.cumsum([a.size for a in W])
    arrays_out['V'] = lda.getV()
    arrays_out['iterations'] = int(dic['iteration']) # name could change
    arrays_out['Z_corpus'] = np.array(np.hstack(Z), dtype=np.int32)
    arrays_out['Z_indices'] = np.cumsum([a.size for a in Z])
    arrays_out['doc_top'] = doc_top
    arrays_out['top_word'] = top_word
    arrays_out['sum_word_top'] = (lda.getV() * float(dic['doc_prior'])) +\
                                  np.zeros(lda.getK())
    arrays_out['context_type'] = ctx_type
    arrays_out['K'] = int(dic['K'])
    arrays_out['alpha'] = float(dic['top_prior'])
    arrays_out['beta'] = float(dic['doc_prior'])
    arrays_out['log_prob_init'] = False
    """
    # log_prob_init = False for now
    dt = [('i', np.int), ('v', np.float)]
    logs = [float(dic['log_probs'])] * int(dic['iteration'])
    indices = range(0, int(dic['iteration']))
    arrays_out['log_prob_init'] = np.array(zip(indices, logs), dtype=dt)
    """
    print 'Saving LDA model to', fname
    np.savez(fname, **arrays_out)
   

def nested_arr_to_np(arr, arrarr=False):
    
    outli = []
    for r in arr:
        inli = []
        for c in r:
            inli.append(c)
        
        if arrarr:
            inli = np.array(inli)    
        outli.append(inli)
    
    return np.array(outli)

    
if __name__=='__main__':
    from vsm.corpus import Corpus
   
    path = '../org/knowceans/gibbstest/'
    c = Corpus.load(path+'church_corp.npz')

    writepath = '/home/doori/inpho/org/knowceans/gibbstest/'
    ctx = 'document'
    # java can't process '..' in the path.
    gw, m = lda_run(c, path+'churchcorp.txt', ctx, 10000, 2, 
                    writepath+'church-meta.txt', 0.01, 0.01)

    save(m, ctx, writepath+'church_lda.npz', writepath+'church-meta.txt')
