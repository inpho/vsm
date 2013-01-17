import numpy as np
from scipy import sparse

from vsm import (row_norms, col_norms, enum_array, enum_sort, 
                 sparse_mvdot, def_submat_size)




# TODO: Suppress division by zero errors; be sure that it's safe to do
# so.



def row_cosines(row, matrix, norms=None,
                filter_nan=False, sort=True,
                submat_size=def_submat_size):
    """
    `row` must be a 2-dimensional array.
    """
    if sparse.issparse(matrix):

        matrix = matrix.tocsr()

        nums = sparse_mvdot(matrix, row.T, submat_size=submat_size)

    else:
        
        nums = np.dot(matrix, row.T)

        nums = np.squeeze(nums)

    if norms is None:

        norms = row_norms(matrix)

    row_norm = row_norms(row)[0]

    dens = norms * row_norm

    out = nums / dens



    if sort:
        
        out = enum_sort(out)

    else:

        out = enum_array(out)
    
    if filter_nan:

        out = out[np.isfinite(out['value'])]

    return out



def simmat_rows(matrix, row_indices):
    """
    """
    sim_matrix = SimilarityMatrix(indices=row_indices)
    
    sim_matrix.compute(matrix)

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

            results = row_cosines(data[:1, :], data, norms=norms, sort=False)

            results = np.array([v for j,v in results])

            self.matrix[i, i:] = results[:]

            data = data[1:, :]

            norms = norms[1:]

        i += 1

        results = row_cosines(data[:1, :], data, norms=norms, sort=False)

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
