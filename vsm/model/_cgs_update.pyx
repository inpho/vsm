# cython: binding=True

import numpy as np
import cython

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def cgs_update_int_char(int itr, 
               unsigned int [:] corpus,
               double [:,:] word_top,
               double [:] inv_top_sums,
               double [:,:] top_doc,
               unsigned char [:] Z,
               int [:] indices,
               str mtrand_str,
               unsigned int [:] mtrand_keys,
               int mtrand_pos,
               int mtrand_has_gauss,
               float mtrand_cached_gaussian):

    cdef int V = corpus.shape[0]
    cdef int N = indices.shape[0]
    cdef int K = word_top.shape[1]

    cdef double log_p = 0
    cdef double [:,:] log_wk = np.log(np.asarray(word_top) * 
                                      np.asarray(inv_top_sums))
    cdef double [:,:] log_kd = np.log(np.asarray(top_doc) /
                                      np.asarray(top_doc).sum(0))

    cdef object np_random_state = np.random.RandomState()
    np_random_state.set_state((mtrand_str, mtrand_keys, 
                               mtrand_pos, mtrand_has_gauss, 
                               mtrand_cached_gaussian))
    cdef double [:] samples = np_random_state.uniform(size=V)
    cdef object mtrand_state = np_random_state.get_state()
    cdef double [:] dist = np.zeros((K,), dtype=np.float64)
    cdef double [:] cum_dist = np.zeros((K,), dtype=np.float64)

    cdef double r, s
    cdef long start, stop, doc_len, offset
    cdef unsigned char k,t
    cdef Py_ssize_t i, j, idx, w

    for i in range(N):

        if i==0:
            doc_len = indices[0]
            offset = 0
        else:
            start = indices[i-1]
            stop = indices[i]
            doc_len = stop - start 
            offset = indices[i-1]

        for j in range(doc_len):

            idx = offset + j
            w,k = corpus[idx], Z[idx]

            log_p += log_wk[w, k] + log_kd[k, i]

            if itr > 0:
                word_top[w, k] -= 1
                s = inv_top_sums[k]
                inv_top_sums[k] = s / (1 - s)
                top_doc[k, i] -= 1

            for t in range(K):
                dist[t] = inv_top_sums[t] * word_top[w,t] * top_doc[t,i]
                if t==0:
                    cum_dist[t] = dist[t]
                else:
                    cum_dist[t] = cum_dist[t-1] + dist[t]
            r = samples[idx] * cum_dist[K-1]
            for k in range(K):
                if r < cum_dist[k]:
                    break

            word_top[w, k] += 1
            s = inv_top_sums[k]
            inv_top_sums[k] = s / (1 + s) 
            top_doc[k, i] += 1

            Z[idx] = k
            
    return (np.asarray(word_top), np.asarray(inv_top_sums), 
            np.asarray(top_doc), np.asarray(Z), log_p, 
            mtrand_state[0], mtrand_state[1], mtrand_state[2], 
            mtrand_state[3], mtrand_state[4])

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def cgs_update_int_short(int itr, 
               unsigned int [:] corpus,
               float [:,:] word_top,
               float [:] inv_top_sums,
               float [:,:] top_doc,
               unsigned short [:] Z,
               int [:] indices,
               str mtrand_str,
               unsigned int [:] mtrand_keys,
               int mtrand_pos,
               int mtrand_has_gauss,
               float mtrand_cached_gaussian):

    cdef int V = corpus.shape[0]
    cdef int N = indices.shape[0]
    cdef int K = word_top.shape[1]

    cdef float log_p = 0
    cdef float [:,:] log_wk = np.log(np.asarray(word_top) * 
                                      np.asarray(inv_top_sums))
    cdef float [:,:] log_kd = np.log(np.asarray(top_doc) /
                                      np.asarray(top_doc).sum(0))

    cdef object np_random_state = np.random.RandomState()
    np_random_state.set_state((mtrand_str, mtrand_keys, 
                               mtrand_pos, mtrand_has_gauss, 
                               mtrand_cached_gaussian))
    cdef float [:] samples = np_random_state.uniform(size=V).astype(np.float32)
    cdef object mtrand_state = np_random_state.get_state()
    cdef float [:] dist = np.zeros((K,), dtype=np.float32)
    cdef float [:] cum_dist = np.zeros((K,), dtype=np.float32)

    cdef float r, s
    cdef long start, stop, doc_len, offset
    cdef unsigned short k, t
    cdef Py_ssize_t i, j, idx, w

    for i in range(N):

        if i==0:
            doc_len = indices[0]
            offset = 0
        else:
            start = indices[i-1]
            stop = indices[i]
            doc_len = stop - start 
            offset = indices[i-1]

        for j in range(doc_len):

            idx = offset + j
            w,k = corpus[idx], Z[idx]

            # float = float, float
            log_p += log_wk[w, k] + log_kd[k, i]

            if itr > 0:
                word_top[w, k] -= 1
                s = inv_top_sums[k]
                inv_top_sums[k] = s / (1 - s)
                top_doc[k, i] -= 1

            for t in range(K):
                #float =  float, float, float
                dist[t] = inv_top_sums[t] * word_top[w,t] * top_doc[t,i]
                if t==0:
                    cum_dist[t] = dist[t]
                else:
                    cum_dist[t] = cum_dist[t-1] + dist[t]
            # float = float, float
            r = samples[idx] * cum_dist[K-1]
            for k in range(K):
                if r < cum_dist[k]:
                    break

            word_top[w, k] += 1
            s = inv_top_sums[k]
            inv_top_sums[k] = s / (1 + s) 
            top_doc[k, i] += 1

            Z[idx] = k
            
    return (np.asarray(word_top), np.asarray(inv_top_sums), 
            np.asarray(top_doc), np.asarray(Z), log_p, 
            mtrand_state[0], mtrand_state[1], mtrand_state[2], 
            mtrand_state[3], mtrand_state[4])

cimport numpy as np
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

cdef extern from "math.h":
    float logf(float n)
cdef inline int int_min(int a, int b): return a if a <= b else b

@cython.binding(True)
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def cgs_update_short_char(int itr, 
               unsigned short [:] corpus,
               np.ndarray[DTYPE_t, ndim=2] word_top,
               np.ndarray[DTYPE_t] inv_top_sums,
               np.ndarray[DTYPE_t, ndim=2] top_doc,
               unsigned char [:] Z,
               int [:] indices,
               str mtrand_str,
               unsigned int [:] mtrand_keys,
               int mtrand_pos,
               int mtrand_has_gauss,
               float mtrand_cached_gaussian):

    cdef int V = corpus.shape[0]
    cdef int N = indices.shape[0]
    cdef int K = word_top.shape[1]
    cdef int W = word_top.shape[0]
    
    cdef Py_ssize_t i, j, idx, w, t
    cdef int first, last

    cdef DTYPE_t log_p = 0
    cdef np.ndarray[DTYPE_t, ndim=2] log_wk = np.log(word_top * inv_top_sums)
    cdef np.ndarray[DTYPE_t, ndim=2] log_kd = np.log(top_doc / top_doc.sum(0))

    cdef object np_random_state = np.random.RandomState()
    np_random_state.set_state((mtrand_str, mtrand_keys, 
                               mtrand_pos, mtrand_has_gauss, 
                               mtrand_cached_gaussian))
    cdef np.ndarray[DTYPE_t] samples = np_random_state.uniform(size=V).astype(DTYPE)
    cdef object mtrand_state = np_random_state.get_state()
    cdef np.ndarray[DTYPE_t] dist = np.zeros((K,), dtype=DTYPE)


    cdef DTYPE_t r, s
    cdef long start, stop, doc_len, offset
    cdef unsigned char k

    with nogil:
        for i in range(N):
    
            if i==0:
                doc_len = indices[0]
                offset = 0
            else:
                start = indices[i-1]
                stop = indices[i]
                doc_len = stop - start 
                offset = indices[i-1]
    
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
                dist[t] = <DTYPE_t>(inv_top_sums[t] * word_top[w,t] * top_doc[t,i])
                for t in range(1,K):
                    dist[t] = dist[t-1] + <DTYPE_t>(inv_top_sums[t] * word_top[w,t] * top_doc[t,i])
                
                r = samples[idx] * dist[K-1]
    
                first = 0
                last = K - 1
                while first < last:
                    k = (first + last) / 2
                    if r < dist[k]:
                        last = k
                    else:
                        first = k + 1
    
                """
                for t in range(K):
                    if r < dist[t]:
                        k = <unsigned char>(t)
                        break
                """
    
                word_top[w, k] += 1
                s = inv_top_sums[k]
                inv_top_sums[k] = s / (1 + s) 
                top_doc[k, i] += 1
    
                Z[idx] = k
            
    return (word_top, inv_top_sums, 
            top_doc, Z, log_p, 
            mtrand_state[0], mtrand_state[1], mtrand_state[2], 
            mtrand_state[3], mtrand_state[4])

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def cgs_update_short_short(int itr, 
               unsigned short [:] corpus,
               double [:,:] word_top,
               double [:] inv_top_sums,
               double [:,:] top_doc,
               unsigned short [:] Z,
               int [:] indices,
               str mtrand_str,
               unsigned int [:] mtrand_keys,
               int mtrand_pos,
               int mtrand_has_gauss,
               float mtrand_cached_gaussian):

    cdef int V = corpus.shape[0]
    cdef int N = indices.shape[0]
    cdef int K = word_top.shape[1]

    cdef double log_p = 0
    cdef double [:,:] log_wk = np.log(np.asarray(word_top) * 
                                      np.asarray(inv_top_sums))
    cdef double [:,:] log_kd = np.log(np.asarray(top_doc) /
                                      np.asarray(top_doc).sum(0))

    cdef object np_random_state = np.random.RandomState()
    np_random_state.set_state((mtrand_str, mtrand_keys, 
                               mtrand_pos, mtrand_has_gauss, 
                               mtrand_cached_gaussian))
    cdef double [:] samples = np_random_state.uniform(size=V)
    cdef object mtrand_state = np_random_state.get_state()
    cdef double [:] dist = np.zeros((K,), dtype=np.float64)
    cdef double [:] cum_dist = np.zeros((K,), dtype=np.float64)

    cdef double r, s
    cdef long start, stop, doc_len, offset
    cdef unsigned short k,t
    cdef Py_ssize_t i, j, idx, w

    for i in range(N):

        if i==0:
            doc_len = indices[0]
            offset = 0
        else:
            start = indices[i-1]
            stop = indices[i]
            doc_len = stop - start 
            offset = indices[i-1]

        for j in range(doc_len):

            idx = offset + j
            w,k = corpus[idx], Z[idx]

            log_p += log_wk[w, k] + log_kd[k, i]

            if itr > 0:
                word_top[w, k] -= 1
                s = inv_top_sums[k]
                inv_top_sums[k] = s / (1 - s)
                top_doc[k, i] -= 1

            for t in range(K):
                dist[t] = inv_top_sums[t] * word_top[w,t] * top_doc[t,i]
                if t==0:
                    cum_dist[t] = dist[t]
                else:
                    cum_dist[t] = cum_dist[t-1] + dist[t]
            r = samples[idx] * cum_dist[K-1]
            for k in range(K):
                if r < cum_dist[k]:
                    break

            word_top[w, k] += 1
            s = inv_top_sums[k]
            inv_top_sums[k] = s / (1 + s) 
            top_doc[k, i] += 1

            Z[idx] = k
            
    return (np.asarray(word_top), np.asarray(inv_top_sums), 
            np.asarray(top_doc), np.asarray(Z), log_p, 
            mtrand_state[0], mtrand_state[1], mtrand_state[2], 
            mtrand_state[3], mtrand_state[4])
