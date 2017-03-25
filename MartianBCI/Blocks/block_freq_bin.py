# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 04:48:50 2017

@author: marzipan
"""

if __name__ == "__main__":
    from Block import Block
else:
    from .Block import Block
import numpy as np

class block_freq_bin (Block):
    '''
    This is meant to feed a classifier or feedback processor with spectral density
    of a particular frequency bin. 
    
    Input: block_inc_spectrogram
    Output: (1 x num_chan) of power for desired bin
    '''
    
    def __init__(self, _pipe, _fs=250, freq_range=(8,13)): 
        assert len(freq_range) == 2, "freq_range must be tuple of length 2"
        self.freq_range = freq_range
        self.pipe = _pipe
        self.fs = _fs
        
    def run(self, inbuf):
        '''
        inbuf will be flattened array of length (129xnum_ch), this method should
        reshape the array into the original shape and extract and sum the coefficients
        corresponding to the desired freq_range_range.
        '''
        lbuf = inbuf.reshape((129,8))
        freqs = np.linspace(0,self.fs/2,129)
        power = np.zeros((8))
        for ch in range(8): #TODO change this to matrix comparison instead of loop
            for f in range(129):
                if freqs[f] >= self.freq_range[0] and freqs[f] <= self.freq_range[1]:
                    power[ch] += lbuf[f,ch]
        return power
    
    def get_output_dim(self, buf_len, chan_sel):
        return 8 #TODO make dynamic
