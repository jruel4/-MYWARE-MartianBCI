#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 07:25:48 2017

@author: marzipan
"""

if __name__ == "__main__":
    from Block import Block
else:
    from .Block import Block
import numpy as np
from MartianBCI.Utils.wavelets import make_morlet

class block_morlet (Block):
    
    def __init__(self, _pipe, f_fwhm=[(12,4)]):  
        '''
        f_fwhm is a list of tuples where each tuple is (frequency, full-width-half-max)
        '''
        self.mPipe = _pipe
        # create wavelet 
        self.wavelets_mat = np.asarray([make_morlet(f,fwhm) for f,fwhm in f_fwhm]).transpose()
        self.num_freqs = len(f_fwhm)
        assert self.wavelets_mat.shape == (250, self.num_freqs)
        
    def run(self, inbuf):
      # Input buf should be a np array of shape
      assert inbuf.shape == (8, 250)
      power = np.abs(np.dot(inbuf, self.wavelets_mat)) # Volts
      return {'default':power}
      
    def get_output_struct(self):
        return {'default' : (8,self.num_freqs)}
      
      
      
      
      
      
      
      