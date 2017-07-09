# -*- coding: utf-8 -*-

"""
Created on Mon Mar 20 00:31:31 2017

@author: marzipan
"""

from .Block import Block
import numpy as np

class block_reshape (Block):
    
    def __init__(self, _pipe, _INPUT_SHAPE, _OUTPUT_SHAPE):   
        self.mPipe = _pipe
        
        #Verify input parameters
        try:
            iter(_INPUT_SHAPE)
            iter(_OUTPUT_SHAPE)
        except TypeError:
            raise TypeError("Input or output shape is not iterable, I: ", _INPUT_SHAPE, "   O:", _OUTPUT_SHAPE)
            
        self.mInputShape = _INPUT_SHAPE
        self.mOutputShape = _OUTPUT_SHAPE
        
        self.mInKeys = None
        
    def run(self, _buf, test=False):
        
        if self.mInKeys == None:
            self.mInKeys = super().get_input_keys(self.mPipe)
        buf = np.asarray(_buf[self.mInKeys[0]])
        return {'reshape':np.reshape(buf, self.mOutputShape)}
    
    def get_output_struct(self):
        print('reshape ',int(np.multiply.reduce(self.mInputShape)))
        return {'reshape':int(np.multiply.reduce(self.mInputShape))}
    
    
    
    
    
    
    
    
    