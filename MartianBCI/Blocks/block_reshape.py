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

#        self.mInKeys = super(block_reshape, self).get_input_keys(self.mPipe)

        self.once = True
        
    def run(self, _buf, test=False):
        buf = super(block_reshape, self).get_default(_buf)
        assert buf.shape == self.mInputShape,\
            "Buffer input shape must match the input shape specified when creating reshape block, got: " +\
                str(buf.shape[0]) + "x" + str(buf.shape[1]) + " expected: " + str(self.mInputShape[0]) + "x" + str(self.mInputShape[1])
        outbuf = np.reshape(buf, self.mOutputShape)
        if self.once:
            print
            print "RS, _buf shape: ", buf.shape
            print "RS, _buf reshape (output) shape: ", outbuf.shape
            print
            self.once = False
        return {'default':outbuf}
    
    def get_output_struct(self):
        print('RS, shape: ',int(np.multiply.reduce(self.mInputShape)))
        return {'default':int(np.multiply.reduce(self.mInputShape))}
    
    
    
    
    
    
    
    
    