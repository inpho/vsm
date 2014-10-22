import unittest2 as unittest
import numpy as np

from vsm import *
from vsm.structarr import *

class TestCore(unittest.TestCase):

    def test_arr_add_field(self):

        arr = np.array([(1, '1'), (2, '2'), (3, '3')],
                   dtype=[('i', np.int), ('c', '|S1')])
        new_arr = np.array([(1, '1', 0), (2, '2', 0), (3, '3', 0)],
                       dtype=[('i', np.int), ('c', '|S1'), ('new', np.int)])

        new_field = 'new'
        vals = np.zeros(3, dtype=np.int)

        test_arr = arr_add_field(arr, new_field, vals)

        self.assertTrue((new_arr==test_arr).all())
        self.assertTrue(new_arr.dtype==test_arr.dtype)

    def test_enum_matrix(self):

        arr = np.array([[6,3,7], [2,0,4]])
        em1 = enum_matrix(arr)
        em2 = enum_matrix(arr, indices=[10,20,30], field_name='tens')

        self.assertTrue((em1 == np.array([[(2,7), (0,6), (1, 3)],[(2,4), (0,2), (1,0)]],
                        dtype=[('i', '<i8'), ('value', '<i8')])).all())
        self.assertTrue((em2 == np.array([[(30,7), (10,6), (20, 3)],[(30,4), (10,2), (20,0)]],
                        dtype=[('tens', '<i8'), ('value', '<i8')])).all())
        


    def test_enum_sort(self):
        
        arr = np.array([7,3,1,8,2])
        sorted_arr = enum_sort(arr)
        sorted_arr1 = enum_sort(arr, indices=[10,20,30,40,50])

        self.assertTrue((sorted_arr == 
            np.array([(3, 8), (0, 7), (1, 3), (4, 2), (2, 1)],
            dtype=[('i', '<i8'), ('value', '<i8')])).all())

        self.assertTrue((sorted_arr1 ==
            np.array([(40, 8), (10, 7), (20, 3), (50, 2), (30, 1)], 
                  dtype=[('i', '<i8'), ('value', '<i8')])).all())


    def test_enum_array(self):
        
        arr1 = np.array([7,3,1,8,2])
        ea1 = enum_array(arr1)
        arr2 = np.array([6,3,7,2,0,4])
        ea2 = enum_array(arr2)

        self.assertTrue((ea1 == 
            np.array([(0,7), (1,3), (2,1), (3,8), (4,2)],
                    dtype=[('i', '<i8'), ('value', '<i8')])).all())
        self.assertTrue((ea2 ==
            np.array([(0,6), (1,3), (2,7), (3,2), (4,0), (5,4)],
                    dtype=[('i', '<i8'), ('value', '<i8')])).all())
        

    def test_zip_arr(self):
        
        arr1 = np.array([[2,4], [6,8]])
        arr2 = np.array([[1,3], [5,7]])

        zipped = zip_arr(arr1, arr2, field_names=['even', 'odd'])
        self.assertTrue((zipped == np.array([[(2,1), (4,3)], [(6,5), (8,7)]],
                        dtype=[('even', '<i8'), ('odd', '<i8')])).all())


    def test_map_strarr(self):

        arr = np.array([(0, 1.), (1, 2.)], 
                   dtype=[('i', 'i4'), ('v', 'f4')])
        m = ['foo', 'bar']
        arr = map_strarr(arr, m, 'i', new_k='str')

        self.assertTrue((arr['str'] == np.array(m, 
                        dtype=np.array(m).dtype)).all())
        self.assertTrue((arr['v'] == np.array([1., 2.], dtype='f4')).all())


suite = unittest.TestLoader().loadTestsFromTestCase(TestCore)
unittest.TextTestRunner(verbosity=2).run(suite)
