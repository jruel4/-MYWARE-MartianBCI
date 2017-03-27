# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 12:03:28 2017

@author: marzipan
"""

if __name__ == "__main__":
    from Block import Block
else:
    from .Block import Block
from collections import deque
import numpy as np

class block_moving_average (Block):
    '''
    Abstract class for Pipeline, indicates required methods
    '''
    def __init__(self, _pipe, _window_len=25):
        self.window_len = _window_len
        self.buf = deque(maxlen=_window_len)
    
    def run(self,_buf):
        '''
        simple moving average
        '''
        self.buf.append(_buf)
        output = np.zeros_like(_buf)
        for b in self.buf:
            output += b
        l = len(self.buf)
        return output/len(self.buf) if l > 0 else output
    
    def get_output_dim(self, buf_len, chan_sel):
        return 8
        
        