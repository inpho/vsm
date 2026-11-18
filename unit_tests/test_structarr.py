import unittest
import numpy as np

from vsm import *
from vsm.structarr import *

class TestCore(unittest.TestCase):

    def test_arr_add_field(self):

        arr = np.array([(1, '1'), (2, '2'), (3, '3')],
                   dtype=[('i', int), ('c', '|S1')])
        new_arr = np.array([(1, '1', 0), (2, '2', 0), (3, '3', 0)],
                       dtype=[('i', int), ('c', '|S1'), ('new', int)])

        new_field = 'new'
        vals = np.zeros(3, dtype=int)

        test_arr = arr_add_field(arr, new_field, vals)

        self.assertTrue(np.array_equiv(new_arr, test_arr))
        self.assertTrue(new_arr.dtype==test_arr.dtype)

    def test_enum_matrix(self):

        arr = np.array([[6,3,7], [2,0,4]], dtype=int)
        em1 = enum_matrix(arr)
        em2 = enum_matrix(arr, 
                          indices=[10,20,30], 
                          field_name='tens')

        self.assertTrue(np.array_equiv(em1, np.array([[(2,7), (0,6), (1, 3)],[(2,4), (0,2), (1,0)]],
                        dtype=[('i', int), ('value', int)])))
        self.assertTrue(np.array_equiv(em2, np.array([[(30,7), (10,6), (20, 3)],[(30,4), (10,2), (20,0)]],
                        dtype=[('tens', int), ('value', int)])))
        


    def test_enum_sort(self):
        
        arr = np.array([7,3,1,8,2])
        sorted_arr = enum_sort(arr)
        sorted_arr1 = enum_sort(arr, indices=[10,20,30,40,50])

        self.assertTrue(np.array_equiv(sorted_arr, 
            np.array([(3, 8), (0, 7), (1, 3), (4, 2), (2, 1)],
            dtype=[('i', int), ('value', int)])))

        self.assertTrue(np.array_equiv(sorted_arr1,
            np.array([(40, 8), (10, 7), (20, 3), (50, 2), (30, 1)], 
                  dtype=[('i', int), ('value', int)])))


    def test_enum_array(self):
        
        arr1 = np.array([7,3,1,8,2])
        ea1 = enum_array(arr1)
        arr2 = np.array([6,3,7,2,0,4])
        ea2 = enum_array(arr2)

        self.assertTrue(np.array_equiv(ea1, 
            np.array([(0,7), (1,3), (2,1), (3,8), (4,2)],
                    dtype=[('i', int), ('value', int)])))
        self.assertTrue(np.array_equiv(ea2,
            np.array([(0,6), (1,3), (2,7), (3,2), (4,0), (5,4)],
                    dtype=[('i', int), ('value', int)])))
        

    def test_zip_arr(self):
        
        arr1 = np.array([[2,4], [6,8]], dtype=int)
        arr2 = np.array([[1,3], [5,7]], dtype=int)

        zipped = zip_arr(arr1, arr2, field_names=['even', 'odd'])
        self.assertTrue(np.array_equiv(zipped, np.array([[(2,1), (4,3)], [(6,5), (8,7)]],
                        dtype=[('even', int), ('odd', int)])))


    def test_map_strarr(self):

        arr = np.array([(0, 1.), (1, 2.)], 
                   dtype=[('i', 'i4'), ('v', 'f4')])
        m = ['foo', 'bar']
        arr = map_strarr(arr, m, 'i', new_k='str')

        self.assertTrue(np.array_equal(arr['str'], 
                        np.array(m, dtype=np.array(m).dtype)))
        self.assertTrue(np.array_equal(arr['v'], np.array([1., 2.], dtype='f4')))


suite = unittest.TestLoader().loadTestsFromTestCase(TestCore)
unittest.TextTestRunner(verbosity=2).run(suite)
