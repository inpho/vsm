import numpy as np


class Model(object):
    """
    """
    def __init__(self, matrix=None):

        self.matrix = matrix
        

    @staticmethod
    def load_matrix(file):
        """
        Loads a matrix that has been stored using `save_matrix`.
        
        Parameters
        ----------
        file : str-like or file-like object
            Designates the file to read. If `file` is a string ending
            in `.gz`, the file is first gunzipped. See `numpy.load`
            for further details.

        Returns
        -------
        A dictionary storing the data found in `file`.

        See Also
        --------
        Model.save
        numpy.load
        """
        print 'Loading matrix from', file

        
        # The slice [()] is for the cases where np.save has stored a
        # sparse matrix in a zero-dimensional array

        return np.load(file)[()]



    def save_matrix(self, file):
        """
        Saves `matrix` from a Model object as an `npz` file.
        
        Parameters
        ----------
        file : str-like or file-like object
            Designates the file to which to save data. See
            `numpy.savez` for further details.
            
        Returns
        -------
        None

        See Also
        --------
        Model.load
        numpy.savez
        """
        print 'Saving matrix as', file
        
        np.save(file, self.matrix)


        
    def train(self, corpus):
        """
        """
        print 'This training function is empty. '\
              'Use a subclass of Model to train a model.'
