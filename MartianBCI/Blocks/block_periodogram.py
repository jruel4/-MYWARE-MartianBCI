# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 00:35:03 2017

@author: marzipan
"""

from Block import Block
import numpy as np
from scipy import signal

class block_periodogram (Block):
    
    def __init__(self, _pipe, _NCHAN, _BUFLEN, nfft=None, window=None):   
        self.mPipe = _pipe
        self.mnChan = _NCHAN
        self.mBufLen = _BUFLEN

        # Number of fft points
        if nfft is None:
            self.mnFFT = _BUFLEN
        elif nfft < _BUFLEN:
            raise ValueError("nFFT must be equal to or greater than the number of input points")
        else:
            self.mnFFT = nfft
            
        # Assign window
        if window is None:
            self.mWindow = signal.tukey(_BUFLEN, 0.25)
        else:
            assert len(window) == _BUFLEN, "Window and buffer length must be the same size"
            self.mWindow = window
        
        # Input buffer
        self.mBuf = np.zeros([_NCHAN, _BUFLEN])

        # Get input keys (not necessary)
#        self.mInKeys = super(block_periodogram, self).get_input_keys(self.mPipe)

        self.once = True
      
    '''
    Expects buf['default'] to be nparray, shape = (nchan, nsamples)
    '''
    def run(self, _buf, test=False):
        buf = super(block_periodogram, self).get_default(_buf)
        assert buf.shape[0] == self.mnChan, "Input dim-0 must be same as number of channels"
        assert buf.shape[1] < self.mBufLen, "Input dim-1 must be less than buffer length"

        in_len = buf.shape[1]

        # shift buffer and add new data to the end
        self.mBuf = np.roll(self.mBuf, -(in_len), 1)
        self.mBuf[:,-(in_len):] = buf
        
        # window and fft
        self.mBufWindowed = self.mBuf * self.mWindow
        fft = np.abs(np.fft.fft(self.mBuf, n=self.mnFFT))

        if self.once:
            print
            print "PER, _buf shape: ", buf.shape
            print "PER, mBuf shape: ", self.mBuf.shape
            print "PER, window shape: ", self.mWindow.shape
            print "PER, windowed buf shape: ", self.mBufWindowed.shape
            print "PER, fft (output) shape: ", fft.shape
            print
            self.once = False
        
        return {'default':fft}
        
    def get_output_struct(self):
        return {'default':[self.mnChan,self.mnFFT]}