import numpy as np


def lda_save(ctx_type, phifile, thetafile, zfile, restfile, modelfile):
    # Reads data from too many files.
    dic = file_to_dict(restfile)
    top_doc = file_to_mat(thetafile)
    word_top = file_to_mat(phifile)

    arrays_out = dict()
    arrays_out['iteration'] = int(dic['iteration'])
    dt = dtype=[('i', np.int), ('v', np.float)]
    logs = [float(dic['log_probs'])] * int(dic['iteration'])
    indices = range(0, int(dic['iteration']))
    arrays_out['log_probs'] = np.array(zip(indices, logs), dtype=dt)
    arrays_out['Z'] = list(file_to_arrli(zfile))
    arrays_out['top_doc'] = top_doc.T
    arrays_out['word_top'] = word_top.T
    arrays_out['inv_top_sums'] = np.array([float(dic['inv_top_sums'])]
                                    * word_top.shape[1])
    arrays_out['context_type'] = ctx_type
    arrays_out['K'] = int(dic['K'])
    arrays_out['m_words'] = int(dic['m_words'])
    arrays_out['doc_prior'] = np.array([float(dic['doc_prior'])]
                                    * top_doc.size)#.reshape(top_doc.shape)
    arrays_out['top_prior'] = np.array([float(dic['top_prior'])]
                                    * word_top.size)#.reshape(word_top.shape)
    
    print 'Saving LDA model to', modelfile
    np.savez(modelfile, **arrays_out)


def file_to_dict(filename):
    """
    Reads a file where each line is 'k,v'
    and returns a dictionary of k,v.
    """
    dic = dict()
    with open(filename, 'r') as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip('\n')
            li = l.split(',')
            dic[li[0]] = li[1]
    return dic


def write_file(corpus, ctx_type, filename):
    """
    Writes corpus.view_contexts(ctx_type) to a file txt.
    """
    ctx = corpus.view_contexts(ctx_type) # [arrays,]

    with open(filename, 'w') as f:
        for arr in ctx:
            for i in arr:
                f.write(str(i))
                f.write('\n')
            f.write('\n')


def file_to_mat(filename):
    """
    Data to an array. works for theta, phi.
    Removes automatically added 'missing values' at the end of the rows.
    """
    arr = np.genfromtxt(filename, delimiter=',')

    return arr[:,:-1]


def file_to_arrli(filename, dtype='int'):
    """
    for Z, list of arrays where each array represents a document
    and the array has topic assignment for each word position in the document.
    Length of the array varies as it depends on the length of the 
    corresponding document.
    """

    with open(filename, 'r') as f:
        lines = f.readlines()

        docs = []
        for l in lines:
            l = l.strip('\n')
            arr = np.fromstring(l, dtype=dtype, sep=',')
            docs.append(arr)

    return docs


