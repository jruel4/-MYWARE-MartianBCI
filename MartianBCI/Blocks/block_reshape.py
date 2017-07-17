# -*- coding: utf-8 -*-

"""
Created on Mon Mar 20 00:31:31 2017

@author: marzipan
"""

from .Block import Block
import numpy as np

class block_reshape (Block):
    
    def __init__(self, _pipe):   
        self.mPipe = _pipe
        

        
        self.mInKeys = None
        
    def run(self, _buf, test=False):
        
        if self.mInKeys == None:
            self.mInKeys = super(block_reshape, self).get_input_keys(self.mPipe)
        buf = np.asarray(_buf[self.mInKeys[0]])
        return {'reshape':np.reshape(buf, self.mOutputShape)}
    
    def get_output_struct(self):
        print('reshape ',int(np.multiply.reduce(self.mInputShape)))
        return {'reshape':int(np.multiply.reduce(self.mInputShape))}
    
    
    
    
    
    
    
    
    