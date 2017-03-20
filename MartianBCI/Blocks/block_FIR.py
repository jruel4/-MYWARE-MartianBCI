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
    
    def __init__(self, fs=250, fir_coeffs=None): 
        self.fs = fs
        if fir_coeffs == None:
            fir_coeffs = self.init_default_filter()
        self.fir_coeffs = fir_coeffs
        self.buf = deque([0 for i in range(len(fir_coeffs))], maxlen=len(fir_coeffs))
        
    def run(self, inbuf):
        self.buf.append(inbuf[0])
        return np.asarray([np.dot(self.fir_coeffs, self.buf)])

    
    def get_output_dim(self, buf_len):
        return 1

    def init_default_filter(self):
        nyq = self.fs/2.0
        desired = (0, 0, 1, 1, 0, 0)
        bands = (0, 1, 2, 30, 31, 125)
        fir_remez = signal.remez(257, bands, desired[::2], Hz=2 * nyq)
        return np.asarray(fir_remez)








