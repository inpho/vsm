import numpy as np



class BaseModel(object):
    """
    """
    def __init__(self, matrix=None):
        """
        """
        self.matrix = matrix
        

    def save(self, f):
        """
        Takes a filename or file object and saves `self.matrix` in an
        npz archive.
        
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
        BaseModel.load
        numpy.savez
        """
        print 'Saving model to', f
        np.savez(f, matrix=self.matrix)


    @staticmethod
    def load(f):
        """
        Takes a filename or file object and loads it as an npz archive
        into a BaseModel object.

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
        BaseModel.save
        numpy.load
        """
        print 'Loading model from', f
        npz = np.load(f)
        
        # The slice [()] is to unwrap sparse matrices, which get saved
        # in singleton object arrays
        return BaseModel(matrix=npz['matrix'][()])
