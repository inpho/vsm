import numpy as np
import cython
@cython.boundscheck(False)
@cython.cdivision(True)


def cgs_update(int itr, 
               int [:] corpus,
               double [:,:] word_top,
               double [:] inv_top_sums,
               double [:,:] top_doc,
               int [:] Z,
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
    cdef Py_ssize_t i, j, idx, w, k, t

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
