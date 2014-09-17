# cython: wraparound=False
# cython: boundscheck=False
# cython: cdivision=True

import numpy as np
cimport cython
cimport openmp
cimport cython.parallel as cp

from libc.stdio cimport printf
from libc.stdlib cimport srand, rand, RAND_MAX, abort, calloc, free



cdef double **calloc_2d(int m, int n) nogil:

    cdef double **a
    cdef int i

    a = <double **> calloc(m, sizeof(double*))
    if a==NULL:
        abort()

    for i in range(m):
        a[i] = <double *> calloc(n, sizeof(double))
        if a[i]==NULL:
            abort()

    return a


cdef void free_2d(double **a, int m) nogil:
    cdef int i
    for i in range(m):
        free(a[i])
    free(a)


cdef int seed_random(unsigned long seed):
    srand(seed)
    return 0


cdef double random() nogil:
    return <double>rand() / <double>RAND_MAX


cdef int init_word_top(int K, 
                       int V,
                       double [:,:] word_top,
                       double [:] beta):
    cdef int w, k
    for k in range(K):
        for w in range(V):
            word_top[w,k] = beta[w]
    return 0


cdef int init_doc_top(int K, 
                      int N,
                      double [:,:] doc_top,
                      double [:] alpha):
    cdef int w, k
    for k in range(K):
        for d in range(N):
            doc_top[d,k] = alpha[k]
    return 0


cdef int sample(int w,
                int i,
                int K,
                double *inv_top_sums,
                double **word_top,
                double **doc_top,
                double *dist,
                double *cdist) nogil:
    cdef int k
    cdef double r

    for k in range(K):
        dist[k] = inv_top_sums[k] * word_top[w][k] * doc_top[i][k]
        if k==0:
            cdist[0] = dist[0]
        else:
            cdist[k] = cdist[k-1] + dist[k]
    r = random() * cdist[K - 1]
    for k in range(K):
       if r <= cdist[k]:
           return k
    abort()
    return 0


cdef int init_cgs(int K,
                  int V,
                  int N,
                  int W,
                  int [:] indices,
                  int [:] corpus,
                  int [:] Z,
                  double [:,:] word_top,
                  double [:,:] doc_top,
                  double [:] inv_top_sums,
                  int [:] Z_cached,
                  double [:,:] word_top_cached,
                  double [:,:] doc_top_cached,
                  double [:] inv_top_sums_cached,
                  double [:] dist,
                  double [:] cdist,
                  double [:] log_probs,
                  int n_threads,
                  int iteration) nogil:

    cdef int i, j, D, start, stop, w, k, ID, u, v, d, dD, dstart, dstop, foo
    cdef double s
    cdef int *Z_local
    cdef double **word_top_local
    cdef double *inv_top_sums_local
    cdef double **doc_top_local
    cdef double *dist_local
    cdef double *cdist_local


    with nogil, cp.parallel(num_threads=n_threads):

        ID = cp.threadid()

        dstart = (ID * N) / n_threads
        if ID == (n_threads-1):
            dstop = N
        else:
            dstop = ((ID + 1) * N) / n_threads

        if ID==0:
            start = 0
            d = indices[0]
        else:
            start = indices[dstart-1]
            d = indices[dstop-1] - start

        Z_local = <int *>calloc(d, sizeof(int))
        if Z_local==NULL:
            abort()

        word_top_local = calloc_2d(V, K)

        inv_top_sums_local = <double *>calloc(K, sizeof(double))
        if inv_top_sums_local==NULL:
            abort()

        dist_local = <double *>calloc(K, sizeof(double))
        if dist_local==NULL:
            abort()

        cdist_local = <double *>calloc(K, sizeof(double))
        if cdist_local==NULL:
            abort()

        doc_top_local = calloc_2d(N, K)

        for u in range(d):
            Z_local[u] = Z[start + u]
        for u in range(V):
            for v in range(K):
                word_top_local[u][v] = word_top[u, v]
        for u in range(K):
            inv_top_sums_local[u] = inv_top_sums[u]
        for u in range(N):
            for v in range(K):
                doc_top_local[u][v] = doc_top[u, v]

        printf('start loop\n')
                
        for i in range(dstart, dstop):

            # printf("start %i doc_len %i ID %i doc %i\n", start, d, ID, i)
        
            for j in range(d):
                w = corpus[start + j]
                k = Z_local[j]

                if not iteration == 0:
                    word_top_local[w][k] -= 1
                    s = inv_top_sums_local[k]
                    inv_top_sums_local[k] = s / (1 - s)
                    doc_top_local[i][k] -= 1

                # k = sample(w,
                #            i,
                #            K,
                #            inv_top_sums_local,
                #            word_top_local,
                #            doc_top_local,
                #            dist_local,
                #            cdist_local)

                word_top_local[w][k] += 1
                s = inv_top_sums_local[k]
                inv_top_sums_local[k] = s / (1 + s)
                doc_top_local[i][k] += 1
                Z_local[j] = k
 
        printf('end loop\n')

        for u in range(d):
            Z[start + u] += Z_cached[start + u] - Z_local[u]

        for u in range(V):
            for v in range(K):
                word_top[u, v] += word_top_cached[u, v] - word_top_local[u][v]

        for u in range(K):
            inv_top_sums[u] += inv_top_sums_cached[u] - inv_top_sums_local[u]

        for u in range(N):
            for v in range(K):
                doc_top[u, v] += doc_top_cached[u, v] - doc_top_local[u][v]

        free(Z_local)
        free_2d(word_top_local, K)
        free(inv_top_sums_local)
        free_2d(doc_top_local, N)
        free(dist_local)
        free(cdist_local)

    return 0


                      
cdef int cgs_loop(int K,
                  int V,
                  int N,
                  int W,
                  int [:] indices,
                  int [:] corpus,
                  int [:] Z,
                  double [:] alpha,
                  double [:] beta,
                  double [:,:] word_top,
                  double [:,:] doc_top,
                  double [:] inv_top_sums,             
                  int [:] Z_cached,
                  double [:,:] word_top_cached,
                  double [:,:] doc_top_cached,
                  double [:] inv_top_sums_cached,
                  double [:] dist,
                  double [:] cdist,
                  double [:] log_probs,
                  int n_iterations,
                  int n_threads,
                  unsigned long seed):

    init_word_top(K, V, word_top, beta)
    init_doc_top(K, N, doc_top, alpha)

    seed_random(seed)

    cdef int itr
    for itr in range(0, n_iterations):
        init_cgs(K,
                 V,
                 N,
                 W,
                 indices,
                 corpus,
                 Z,
                 word_top,
                 doc_top,
                 inv_top_sums,
                 Z_cached,
                 word_top_cached,
                 doc_top_cached,
                 inv_top_sums_cached,
                 dist,
                 cdist,
                 log_probs,
                 n_threads,
                 itr)
        printf('Iteration %i complete: log_prob=%f\n', itr, log_probs[itr])

    return 0


def cgs(long K, 
        long V,
        int [:] indices,
        int [:] corpus,
        double [:] alpha,
        double [:] beta,
        long n_iterations,
        long n_threads,
        unsigned long seed):

    cdef int N = indices.shape[0]
    cdef int W = corpus.shape[0]

    cdef int [:] Z = np.zeros((W,), dtype=np.dtype('i'))
    cdef double [:,:] word_top = np.zeros((V,K), dtype=np.dtype('d')) 
    cdef double [:,:] doc_top = np.zeros((N,K), dtype=np.dtype('d'))
    cdef double [:] inv_top_sums = np.zeros((K,), dtype=np.dtype('d'))
    cdef int [:] Z_cached = Z.copy()
    cdef double [:,:] word_top_cached = word_top.copy()
    cdef double [:,:] doc_top_cached = doc_top.copy()
    cdef double [:] inv_top_sums_cached = inv_top_sums.copy()
    cdef double [:] dist = np.zeros((K,), dtype=np.dtype('d'))
    cdef double [:] cdist = np.zeros((K,), dtype=np.dtype('d'))
    cdef double [:] log_probs = np.zeros((n_iterations,), dtype=np.dtype('d'))

    cgs_loop(<int>K,
             V,
             N,
             W,
             indices,
             corpus,
             Z,
             alpha,
             beta,
             word_top,
             doc_top,
             inv_top_sums,
             Z_cached,
             word_top_cached,
             doc_top_cached,
             inv_top_sums_cached,
             dist,
             cdist,
             log_probs,
             <int>n_iterations,
             <int>n_threads,
             seed)

    return { 'Z': np.asarray(Z), 
             'word_top': np.asarray(word_top), 
             'top_doc': np.asarray(doc_top).T,
             'log_probs': np.asarray(log_probs)}
