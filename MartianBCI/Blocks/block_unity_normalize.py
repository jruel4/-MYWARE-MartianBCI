# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 12:02:51 2017

@author: marzipan
"""

if __name__ == "__main__":
    from Block import Block
else:
    from .Block import Block
import numpy as np
    
class block_unity_normalize (Block):
    '''
    Abstract class for Pipeline, indicates required methods
    '''
    def __init__(self, _pipe, _min_x=0.0, _max_x=100.0, _min_y=0.0, _max_y=1.0):
        self.min_x = _min_x
        self.max_x = _max_x
        self.min_y = _min_y
        self.max_y = _max_y
        # construct linear mapping
        self.slope = (self.max_y-self.min_y)/(self.max_x-self.min_x)
    
    def run(self, buf):
        '''
        purpose: ensure that output is mapped to values between 0 and 1
        
        an arbitrary linear mapping could easily compress relevant changes into 
        a small range such that fluctuations in real band power are not easily 
        detected by feedback. Therefore the key is to 'zoom' dynamically into 
        a given (linear) scale.
        '''
        output = np.zeros_like(buf)
        for i in range(len(buf)):
            if buf[i] >= self.max_x:
                output[i] = self.max_y
            elif buf[i] <= self.min_x:
                output[i] = self.min_y
            else:
                output[i] = self.slope * buf[i] + self.min_y
        return output
            
            
    
    def get_output_dim(self, buf_len, chan_sel):
        return 8
    
    