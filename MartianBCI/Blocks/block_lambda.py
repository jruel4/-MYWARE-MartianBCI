# -*- coding: utf-8 -*-

"""
Created on Mon Mar 20 00:31:31 2017

@author: marzipan
"""

from .Block import Block
import numpy as np

class block_lambda (Block):
    
    def __init__(self, _pipe, _LAMBDA_FUNC, _OUTPUT_SHAPE, _PARENT_KEY=None):   
        self.mPipe = _pipe
        
        self.mLambda = _LAMBDA_FUNC
        self.mOutputShape = _OUTPUT_SHAPE
        self.mParentKey = _PARENT_KEY

#        self.mInKeys = super(block_lambda, self).get_input_keys(self.mPipe)

        self.once = True
        
    def run(self, _buf, test=False):
        if self.mParentKey == None:
            buf = super(block_lambda, self).get_default(_buf)
        else:
            buf = _buf[self.mParentKey]

        outbuf = self.mLambda(buf)
        if self.once:
            print
            print "LAM, buf shape: ", buf.shape
            print "LAM, outbuf shape: ", outbuf.shape
            print "LAM, outbuf: ", outbuf
            print
            self.once = False
        return {'default':np.asarray([outbuf])}
    
    def get_output_struct(self):
        print('LAM, shape: ',self.mOutputShape)
        return {'default':self.mOutputShape}
    
    
    
    
    
    
    
    
    