import numpy as np
cimport numpy as np
import cython
@cython.boundscheck(False)
@cython.cdivision(True)

def cgs_update(int itr, 
               np.ndarray[int, negative_indices=False] corpus,
               np.ndarray[np.float64_t, negative_indices=False, ndim=2] word_top, 
               np.ndarray[np.float64_t, negative_indices=False] inv_top_sums,
               np.ndarray[np.float64_t, negative_indices=False, ndim=2] top_doc,
               np.ndarray[long, negative_indices=False] Z,
               np.ndarray[long, negative_indices=False] indices):

    cdef int V = corpus.shape[0]
    cdef int N = indices.shape[0]
    cdef int K = word_top.shape[1]

    cdef double log_p = 0
    cdef np.ndarray[np.float64_t, ndim=2, negative_indices=False, mode='c']\
        log_wk = np.log(word_top * inv_top_sums[np.newaxis, :])
    cdef np.ndarray[np.float64_t, ndim=2, negative_indices=False, mode='c']\
        log_kd = np.log(top_doc / top_doc.sum(0)[np.newaxis, :])

    cdef np.ndarray[np.float64_t, negative_indices=False, mode='c']\
        samples = np.random.random(V)

    cdef double r, s
    cdef long start, stop, doc_len, offset
    cdef Py_ssize_t i, j, idx, w, k, t
    cdef np.ndarray[np.float64_t, ndim=1, negative_indices=False, mode='c']\
        dist = np.zeros((K,), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1, negative_indices=False, mode='c']\
        cum_dist = np.zeros((K,), dtype=np.float64)

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

    return word_top, inv_top_sums, top_doc, Z, log_p
