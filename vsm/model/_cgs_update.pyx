# cython: binding=True
# cython: wraparound=False
# cython: boundscheck=False
# cython: cdivision=True

import cython
cimport cython

import numpy as np
cimport numpy as np

ctypedef np.float32_t NP_FLOAT_t
#TODO: figure out how to use np types in python code
#ctypedef fused NP_FLOAT_t:
#    np.float32_t
#    np.float64_t

ctypedef fused CORPUS_t:
    unsigned int
    unsigned short

ctypedef fused TOPIC_t:
    unsigned short
    unsigned char

cdef extern from "math.h":
    float logf(float n)

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def cgs_update(int itr, 
               CORPUS_t [:] corpus,
               np.ndarray[NP_FLOAT_t, ndim=2] word_top,
               np.ndarray[NP_FLOAT_t] inv_top_sums,
               np.ndarray[NP_FLOAT_t, ndim=2] top_doc,
               TOPIC_t [:] Z,
               int [:] indices,
               str mtrand_str,
               unsigned int [:] mtrand_keys,
               int mtrand_pos,
               int mtrand_has_gauss,
               float mtrand_cached_gaussian):
    
    cdef int first, last
    cdef long stop, doc_len, offset
    cdef NP_FLOAT_t r, s
    cdef Py_ssize_t i, j, idx, w, t, k

    cdef int V = corpus.shape[0]
    cdef int N = indices.shape[0]
    cdef int K = word_top.shape[1]
    cdef int W = word_top.shape[0]
    
    cdef NP_FLOAT_t log_p = 0
    cdef np.ndarray[NP_FLOAT_t, ndim=2] log_wk = np.log(word_top * inv_top_sums)
    cdef np.ndarray[NP_FLOAT_t, ndim=2] log_kd = np.log(top_doc / top_doc.sum(0))

    cdef object np_random_state = np.random.RandomState()
    np_random_state.set_state((mtrand_str, mtrand_keys, 
                               mtrand_pos, mtrand_has_gauss, 
                               mtrand_cached_gaussian))
    cdef np.ndarray[NP_FLOAT_t] samples = np_random_state.uniform(size=V).astype(np.float32)
    cdef np.ndarray[NP_FLOAT_t] dist = np.zeros((K,), dtype=np.float32)

    cdef object mtrand_state = np_random_state.get_state()


    with nogil:
        for i in range(N):
    
            if i==0:
                doc_len = indices[0]
                offset = 0
            else:
                offset = indices[i-1]
                stop = indices[i]
                doc_len = stop - offset 
    
            for j in range(doc_len):
    
                idx = offset + j
                w,k = corpus[idx], Z[idx]
    
                log_p += log_wk[w, k] + log_kd[k, i]
    
                if itr > 0:
                    word_top[w, k] -= 1
                    s = inv_top_sums[k]
                    inv_top_sums[k] = s / (1 - s)
                    top_doc[k, i] -= 1
    
                t = 0
                dist[t] = <NP_FLOAT_t>(inv_top_sums[t] * word_top[w,t] * top_doc[t,i])
                for t in range(1,K):
                    dist[t] = dist[t-1] + <NP_FLOAT_t>(inv_top_sums[t] * word_top[w,t] * top_doc[t,i])
                
                r = samples[idx] * dist[K-1]
                for k in range(K):
                    if r < dist[k]:
                        break
                """
                # This code implements binary search for the right insertion
                # point for the probability in the cumulative distribution
                first = 0
                last = K - 1
                while first < last:
                    k = (first + last) / 2
                    if r < dist[k]:
                        last = k
                    else:
                        first = k + 1
                """
    
                word_top[w, k] += 1
                s = inv_top_sums[k]
                inv_top_sums[k] = s / (1 + s) 
                top_doc[k, i] += 1
    
                Z[idx] = <TOPIC_t>(k)
            
    return (np.asarray(word_top), np.asarray(inv_top_sums), 
            np.asarray(top_doc), np.asarray(Z), log_p, 
            mtrand_state[0], mtrand_state[1], mtrand_state[2], 
            mtrand_state[3], mtrand_state[4])
