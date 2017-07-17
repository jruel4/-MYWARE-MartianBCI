#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 05:32:34 2017

@author: marzipan
"""

# -*- coding: utf-8 -*-
"""
Possible approached to high amplitude noise rejection.

1) discretely epoched, such that there is a definite extra delay...

2) continuous ... perhaps just replacing with average?

We should focus on a specific pipeline, and clean for that 
"""

from scipy import signal
from .Block import Block
import numpy as np

class block_noise_reject (Block):
    
    def __init__(self, _pipe):   
        self.mPipe = _pipe
        
        self.NOISE_THRESH = 150e-6 # Volts
        self.B2V = 4.5/(2**23-1)/24.0
        
    def run(self, inbuf):
      
      # Input buf should be a deque of 250 samples
      d = np.asarray(inbuf)
      assert d.shape == (250,8)
      
      # convert to volts
      d *= self.B2V
      
      # get min, max of each column to see if artifact
      mins = d.min(axis=0)
      maxs = d.max(axis=0)
      
      valid = (maxs - mins) < self.NOISE_THRESH
      
      return {'valid': valid,
              'data': d}
      
    def get_output_struct(self):
        return {'filtered':8}
      
    def set_filter(self, f1, f2):
      self.fir_coeffs = signal.firwin(self.n_taps, [f1, f2], pass_zero=False, nyq=125.0)
    
    
    
    
    
    
    