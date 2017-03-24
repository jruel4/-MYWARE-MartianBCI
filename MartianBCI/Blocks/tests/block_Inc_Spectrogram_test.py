# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 08:55:33 2017

@author: marzipan
"""

import unittest
import numpy as np
from collections import deque
from MartianBCI.Blocks.block_Inc_Spectrogram import block_inc_spectrogram

class TestStringMethods(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Initialize spec block
        cls.num_ch = 8
        cls.spec_block = block_inc_spectrogram(None, fs=250, nperseg=256, num_ch=cls.num_ch)
        
        
    @classmethod
    def tearDownClass(cls):
        del cls.spec_block
        
    def sin_test(self):
        '''
        generate a deque of sine wave and verify output... possible fft run on wrong dimension...
        '''
        pass

    def test_spec_block(self):
        # buf passed in should be deque of 4 lists, with each list num_chan long
        buf = deque([np.ones((self.num_ch)) for i in range(4)])
        result = self.spec_block.run(buf)
        print("result.shape: ",result.shape)
        # Redundant check checks to demo assert options
        self.assertEqual(result.shape, (128,self.num_ch))
        self.assertTrue(result.shape == (128,self.num_ch))
        self.assertFalse(result.shape != (128,self.num_ch))

if __name__ == '__main__':
    unittest.main()
    
    
    
    
    
    
    
    
    
    
    
    

