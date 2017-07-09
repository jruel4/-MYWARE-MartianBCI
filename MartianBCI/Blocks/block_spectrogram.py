# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 00:31:31 2017

@author: marzipan
"""

from scipy import signal
from .Block import Block
import numpy as np

class block_spectrogram (Block):
    
    def __init__(self, _pipe, _INPUT_SHAPE, fs=250, nperseg=None, noverlap=None,nfft=None):   
        self.mPipe = _pipe
        self.mFS = fs
        self.mnOverlap = noverlap
        self.mnPerSeg = nperseg
        self.mnFFT = nfft
        
        #Create out input buffer
        if not isinstance(_INPUT_SHAPE, list):
            raise TypeError("Input shape must be of type list")
        self.mInputShape = _INPUT_SHAPE
        self.mBuf = np.zeros(self.mInputShape)
        
        # JCR - What should this do?
        self.captured_axes = False
        
    def run(self, _buf, test=False):
        
        for i in range(len(_buf)):
            #remove the oldest value from the buffer
            self.mBuf = np.delete(self.mBuf,0,1)
            #and insert the next (& format input to be #chan x #samples)
            _buf_in = np.asarray([[x] for x in _buf.pop()])
            self.mBuf = np.append(self.mBuf,_buf_in,1)

        f, t, Sxx = signal.spectrogram(
                np.asarray(self.mBuf),
                fs=self.mFS,
                nperseg=self.mnPerSeg,
                noverlap=self.mnOverlap,
                nfft=self.mnFFT,
                axis=1)
        if not self.captured_axes and not test:
            self.f = f
            self.t = t
            self.captured_axes = True
#        print("SXX: ", Sxx.shape)
        return {'spectro':Sxx}
    
    def get_axes(self):
        return self.f, self.t
    
    def spectrogram_ready(self):
        return self.captured_axes
    
    def get_output_struct(self):
        return {'spectro':-1}
    
    
    
    
    
    
    
    
    