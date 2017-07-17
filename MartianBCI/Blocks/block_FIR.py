# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 07:35:17 2017

@author: marzipan
"""

import numpy as np
from collections import deque
from scipy import signal

if __name__ == "__main__":
    from Block import Block
else:
    from .Block import Block

class block_fir_filter (Block):
    
    def __init__(self, _pipe, fs=250, fir_coeffs=None): 
        self.pipe = _pipe
        self.fs = fs
        if fir_coeffs == None:
            fir_coeffs = self.init_default_filter()
            self.update_fir_coeffs('Alpha_Bandpass.npz')
        self.fir_coeffs = fir_coeffs
        self.buf = deque([np.zeros((len(_pipe.chan_sel))) for i in range(len(fir_coeffs))], maxlen=len(fir_coeffs))
        
    def run(self, inbuf):
        self.buf.append(inbuf.pop())
        return np.dot(self.fir_coeffs, np.asarray(self.buf))
    
    def get_output_dim(self, buf_len, chan_sel):
        return buf_len * len(chan_sel)

    def init_default_filter(self):
        nyq = self.fs/2.0
        desired = (0, 0, 1, 1, 0, 0)
        bands = (0, 1, 2, 30, 31, 125)
        fir_remez = signal.remez(257, bands, desired[::2], Hz=2 * nyq)
        return np.asarray(fir_remez)

    def update_fir_coeffs(self, _fir_coeffs):
        if type(_fir_coeffs) == str:
            fil = np.load('../MartianBCI/Blocks/Filter_Coefficients/FIR/'+_fir_coeffs)
            _fir_coeffs = fil['ba'][0][:257]
        else:
            assert len(_fir_coeffs) == len(self.fir_coeffs), "new coefficients must be same length as current"
        self.fir_coeffs = _fir_coeffs






