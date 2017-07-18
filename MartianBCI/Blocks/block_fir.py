# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 23:32:28 2017

@author: marzipan
"""

from scipy import signal
from Queue import deque
import numpy as np

if __name__ == "__main__":
    from Block import Block
else:
    from .Block import Block


class block_fir (Block):
    
    def __init__(self, _pipe, _CHANNELS):   
        self.mPipe = _pipe
        self.mnChan = _CHANNELS
        
        self.n_taps = 250
        self.mBufLen = self.n_taps
        self.fir_coeffs = signal.firwin(self.n_taps, [16.0, 20.0], pass_zero=False, nyq=125.0)
        
        self.mBuf = np.zeros([_CHANNELS,self.n_taps])
        
        # Get input keys (not necessary)
#        self.mInKeys = super(block_periodogram, self).get_input_keys(self.mPipe)

        self.once = True
        
    def run(self, _buf):
        buf = super(block_fir, self).get_default(_buf)
        assert buf.shape[0] == self.mnChan, "Input dim-0 must be same as number of channels"
        assert buf.shape[1] < self.mBufLen, "Input dim-1 must be less than buffer length"

        in_len = buf.shape[1]

        # shift buffer and add new data to the end
        self.mBuf = np.roll(self.mBuf, -(in_len), 1)
        self.mBuf[:,-(in_len):] = buf

        self.mBufDotprod = np.dot(self.mBuf, self.fir_coeffs) #dot returns (nchan,), needs to be (nchan,nsamples), see next line
        self.mBufDotprodRS = np.reshape(self.mBufDotprod, [self.mnChan,1]) # should break if not the correct shape
        
        if self.once:
            print
            print "FIR, _buf shape: ", buf.shape
            print "FIR, mBuf shape: ", self.mBuf.shape
            print "FIR, fircoeffs shape: ", self.fir_coeffs.shape
            print "FIR, mBuf dot producted shape: ", self.mBufDotprod.shape
            print "FIR, dotprod shape: ", self.mBufDotprodRS.shape
            print
            self.once = False

        return {'default': self.mBufDotprodRS }
    
    def get_output_struct(self):
        return {'default':self.mnChan}
      
    def set_filter(self, f1, f2):
      self.fir_coeffs = signal.firwin(self.n_taps, [f1, f2], pass_zero=False, nyq=125.0)
    
    
    
    
    
    
    