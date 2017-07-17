#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 07:25:48 2017

@author: marzipan
"""

from scipy import signal
from .Block import Block
import numpy as np

class block_morlet (Block):
    
    def __init__(self, _pipe):   
        self.mPipe = _pipe
        
        # create wavelet 
        M = 250 #TODO this must match epoc len, make this dynamic
        r = 250.0
        s = 1.0/2
        w = 9.2*2
        f = 2*s*w*r/M
        bm = sig.morlet(M, w=w, s=s, complete=True)
        bm /= sum(np.abs(bm))
        
    def run(self, inbuf):
      
      # Input buf should be a np array of shape
      assert inbuf.shape == (250, 8)
      pass
      
    def get_output_struct(self):
        return {'filtered':8}
      
    def set_filter(self, f1, f2):
      self.fir_coeffs = signal.firwin(self.n_taps, [f1, f2], pass_zero=False, nyq=125.0)



def calc_w(f, s):
  srate = 250.0
  M = 250
  # f = 2*s*w*srate/M
  # f/2/srate*M = s*w
  w = f/2/srate*M/s
  return w

def make_morlet(f,s):
  # create wavelet 
  M = 250 #TODO this must match epoc len, make this dynamic
  srate = 250.0  
  w = calc_w(f,s)  
  f = 2*s*w*srate/M
  print("morlet freq: ",f)
  bm = signal.morlet(M, w=w, s=s, complete=True)
  bm /= sum(np.abs(bm))    
  return bm

def calc_fwhm(x):
  freqs = np.linspace(0,250,num=len(x))
  fft = np.abs(np.fft.fft(x))
  amax = np.argmax(fft)
  hmax = fft[amax]/2.0
  fft -= hmax   
  amin = np.argmin(abs(fft))  
  fwhm = abs(freqs[amax] - freqs[amin])*2  
  return fwhm
  

bm = make_morlet(18, 5)
print(calc_fwhm(bm))













