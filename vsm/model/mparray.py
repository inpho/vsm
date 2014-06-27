import multiprocessing as mp

import numpy as np


__all__ = ['mp_shared_array', 'mp_shared_value']




# 
# helper functions for writing numpy arrays to multiprocessing arrays
# 


def mp_shared_array(arr, ctype='i'):
    
    shared_arr_base = mp.Array(ctype, arr.size)
    shared_arr_base[:] = arr
    return np.ctypeslib.as_array(shared_arr_base.get_obj())


def mp_shared_value(x, ctype='i'):

    return mp.Value(ctype, x)
