from numpy import log2, sqrt, sum

def entropy(p):

    return -sum(p * log2(p))

def hellinger(p, q):

    return sqrt(sum((sqrt(p) - sqrt(q))**2))

def kl(p, q):

    return sum(p * log2(p / q))

def jensen_shannon(p, q):

    m = 0.5 * (p + q)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

def euclidean(p, q):

   return sqrt(sum((p - q)**2))
