# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 00:25:35 2017

@author: marzipan
"""

from .Block import Block
import numpy as np
class test_block (Block):
        
        def __init__(self, pos1, pos2, kw1 = "val1", kw2 = "val2"):
            print("pos1: ",pos1)
            print("pos2: ",pos2)
            print("kw1: ",kw1)
            print("kw2: ",kw2)
            self.idx = 0
            
        def run(self, buf):
            self.idx += 1
            return {'data':[x*self.idx for x in range(8)]}
        def get_output_struct(self):
            return {'data': 8}
        def get_self(self):
            return self