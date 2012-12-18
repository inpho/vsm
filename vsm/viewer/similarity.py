import numpy as np
from scipy import sparse

from vsm import row_norms, col_norms, enum_array, enum_sort

def_submat_size = 1e5



# TODO: Suppress division by zero errors; be sure that it's safe to do
# so.



def similar_rows(row_index,
                 matrix,
                 filter_nan=False,
                 sort=True,
                 norms=None,
                 submat_size=def_submat_size):
    """
    """
    if sparse.issparse(matrix):

        matrix = matrix.tocsr()

        nums = sparse_mvdot(matrix,
                            matrix[row_index:row_index+1, :].T,
                            submat_size=submat_size)

    else:
        
        nums = np.dot(matrix, matrix[row_index:row_index+1, :].T)

        nums = np.squeeze(nums)

    if norms is None:

        norms = row_norms(matrix)

    dens = norms * norms[row_index]

    out = nums / dens



    if sort:
        
        out = enum_sort(out)

    else:

        out = enum_array(out)
    
    if filter_nan:

        out = out[np.isfinite(out['v'])]

    return out



def sparse_mvdot(m, v, submat_size=def_submat_size):
    """
    For sparse matrices. The expectation is that a dense view of the
    entire matrix is too large. So the matrix is split into
    submatrices (horizontal slices) which are converted to dense
    matrices one at a time.
    """

    m = m.tocsr()

    if sparse.issparse(v):

        v = v.toarray()

    v = v.reshape((v.size, 1))

    out = np.empty((m.shape[0], 1))

    if submat_size < m.shape[1]:

        print 'Note: specified submatrix size is '\
              'less than the number of columns in matrix'

        m_rows = 1

        k_submats = m.shape[0]

    elif submat_size > m.size:

        m_rows = m.shape[0]

        k_submats = 1

    else:

        m_rows = int(submat_size / m.shape[1])

        k_submats = int(m.shape[0] / m_rows)

    for i in xrange(k_submats):

        i *= m_rows

        j = i + m_rows

        submat = m[i:j, :]

        out[i:j, :] = np.dot(submat.toarray(), v)

    if j < m.shape[0]:

        submat = m[j:, :]

        out[j:, :] = np.dot(submat.toarray(), v)

    return np.squeeze(out)



def similar_columns(col_index,
                    matrix,
                    filter_nan=False,
                    sort=True,
                    norms=None,
                    submat_size=def_submat_size):
    """
    """
    return similar_rows(col_index, matrix.T, filter_nan=filter_nan,
                        sort=sort, norms=norms, submat_size=submat_size)



def simmat_rows(matrix, row_indices):
    """
    """
    sim_matrix = SimilarityMatrix(indices=row_indices)
    
    sim_matrix.compute(matrix)

    return sim_matrix



def simmat_columns(matrix, column_indices):
    """
    """
    sim_matrix = SimilarityMatrix(indices=column_indices)

    sim_matrix.compute(matrix.T)

    return sim_matrix




# TODO: Compress symmetric similarity matrix. Cf. scipy.spatial.distance.squareform
class SimilarityMatrix(object):

    def __init__(self, indices=None, labels=None, matrix=None):

        self.indices = indices
        
        self.labels = labels

        if matrix is None:

            self.matrix = np.zeros((len(self.indices), len(self.indices)))



    def compute(self, data):
        """
        Comparisons are row-wise.

        Returns an upper triangular matrix.
        """
        if sparse.issparse(data):

            data = data.tocsr()
        
        data = data[self.indices]

        norms = row_norms(data)

        for i in xrange(data.shape[0] - 1):

            results = similar_rows(0 , data, norms=norms, sort=False)

            results = np.array([v for j,v in results])

            self.matrix[i, i:] = results[:]

            data = data[1:, :]

            norms = norms[1:]

        i += 1

        results = similar_rows(0 , data, norms=norms, sort=False)

        results = np.array([v for j,v in results])

        self.matrix[i, i:] = results[:]



def test_simmat():

    m = np.random.random((10,5))

    out_1 = np.zeros((10,10))

    for i, j in zip(*np.triu_indices_from(out_1)):

        out_1[i, j] = (np.dot(m[i], m[j])
                       / (np.dot(m[i], m[i])**.5
                          * np.dot(m[j], m[j])**.5))

    out_2 = SimilarityMatrix(indices = range(10))
    
    out_2.compute(m)

    out_2 = out_2.matrix
        
    assert np.allclose(out_1, out_2)
