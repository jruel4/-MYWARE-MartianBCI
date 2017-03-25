# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 05:36:58 2017

@author: marzipan
"""

import unittest
import numpy as np
from MartianBCI.Blocks.block_freq_bin import block_freq_bin

class TestStringMethods(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Initialize spec block
        cls.num_ch = 8
        cls.freq_bin_block = block_freq_bin(None)
        
        
    @classmethod
    def tearDownClass(cls):
        pass
        
    def sin_test(self):
        '''
        generate a deque of sine wave and verify output... possible fft run on wrong dimension...
        '''
        pass

    def test_spec_block(self):
        # buf passed in should be deque of 4 lists, with each list num_chan long
        buf = np.ones((129*8))
        result = self.freq_bin_block.run(buf)
        print("result.shape: ",result.shape)
        # Redundant check checks to demo assert options
        self.assertEqual(result.shape, (8,))

if __name__ == '__main__':
    unittest.main()
    