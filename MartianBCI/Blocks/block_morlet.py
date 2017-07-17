#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 07:25:48 2017

@author: marzipan
"""

from .Block import Block
import numpy as np
from MartianBCI.Utils.wavelets import make_morlet

class block_morlet (Block):
    
    def __init__(self, _pipe):   
        self.mPipe = _pipe
        # create wavelet 
        f = 12.0 # frequency (Hz)
        fwhm = 4 # must be 2,4 or 6
        self.bm = make_morlet(f,fwhm)
        
    def run(self, inbuf):
      # Input buf should be a np array of shape
      assert inbuf.shape == (250, 8)
      power = np.abs(np.dot(self.bm,inbuf)) # Volts
      return {'default':power}
      
    def get_output_struct(self):
        return {'filtered':8}
      