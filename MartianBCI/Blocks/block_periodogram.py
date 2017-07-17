# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 00:35:03 2017

@author: marzipan
"""

from .Block import Block
import numpy as np
from scipy import signal

class block_periodogram (Block):
    
    def __init__(self, _pipe, _INPUT_SHAPE, nfft=None, window=None):   
        self.mPipe = _pipe
        if nfft == None:
            nfft = _INPUT_SHAPE[1] #length of input signal
        elif nfft < _INPUT_SHAPE[1]:
            raise Exception("nFFT must be equal to or greater than the number of input points")

        self.mnFFT = nfft
        
        #Create out input buffer
        if not isinstance(_INPUT_SHAPE, list):
            raise TypeError("Input shape must be of type list")
        self.mInputShape = _INPUT_SHAPE
        self.mBuf = np.zeros(self.mInputShape)
        
    def run(self, _buf, test=False):
        
        for i in range(len(_buf)):
            #remove the oldest value from the buffer
            self.mBuf = np.delete(self.mBuf,0,1)
            #and insert the next (& format input to be #chan x #samples)
            _buf_in = np.asarray([[x] for x in _buf.pop()])
            self.mBuf = np.append(self.mBuf,_buf_in,1)

        fft = np.abs(np.fft.fft(self.mBuf, n=self.mnFFT))
#        print("FFT: ", fft.shape)
        return {'periodo':fft}
        
    def get_output_struct(self):
        return {'periodo':[self.mInputShape[0],self.mnFFT]}
    