# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 23:32:28 2017

@author: marzipan
"""

from scipy import signal
from .Block import Block
from multiprocessing import deque
import numpy as np

class block_fir (Block):
    
    def __init__(self, _pipe):   
        self.mPipe = _pipe
        
        self.n_taps = 250
        self.fir_coeffs = signal.firwin(self.n_taps, [16.0, 20.0], pass_zero=False, nyq=125.0)
        self.buf = deque([np.zeros((len(_pipe.chan_sel))) for i in range(self.n_taps)], maxlen=self.n_taps)
        
    def run(self, inbuf):
        for i in range(len(inbuf)):
          self.buf.append(inbuf.pop())
        return np.dot(self.fir_coeffs, np.asarray(self.buf))
    
    def get_output_struct(self):
        return {'filtered':8}
      
    def set_filter(self, f1, f2):
      self.fir_coeffs = signal.firwin(self.n_taps, [f1, f2], pass_zero=False, nyq=125.0)
    
    
    
    
    
    
    