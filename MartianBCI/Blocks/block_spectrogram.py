# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 00:31:31 2017

@author: marzipan
"""

from scipy import signal
from .Block import Block
import numpy as np

class block_spectrogram (Block):
    
    def __init__(self, _pipe, fs=250, nperseg=256, noverlap=0):   
        self.pipe = _pipe
        self.fs = fs
        self.noverlap = noverlap
        self.nperseg = nperseg
        self.captured_axes = False
        
    def run(self, buf, test=False):
        f, t, Sxx = signal.spectrogram(np.asarray(buf), fs=self.fs, nperseg=self.nperseg, noverlap=self.noverlap)
        if not self.captured_axes and not test:
            self.f = f
            self.t = t
            self.captured_axes = True
        return Sxx.ravel()
    
    def get_axes(self):
        return self.f, self.t
    
    def spectrogram_ready(self):
        return self.captured_axes
    
    def get_output_dim(self, buf_len, chan_sel):
        sample_buf = np.random.randn(buf_len)
        return len(self.run(sample_buf, test=True))
    
    
    
    
    
    
    
    
    