#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 19:31:23 2017

@author: marzipan
"""

from scipy import signal
import numpy as np
from matplotlib import pyplot as plt

fwhm_dict = {2:.6,
             4:1.0,
             6:1.2}


def calc_w(f, s,M=250.0):
  srate = 250.0
  # f = 2*s*w*srate/M
  # f/2/srate*M = s*w
  w = f/2.0/srate*M/s
  return w

def make_morlet(f,fwhm=4,srate=250.0,M=250.0,s=None, debug=False, norm=None):
  '''
  fwhm = full width half max in hz, must be 2, 4 or 6.
  '''
  
  if not s:
      s = fwhm_dict[fwhm]
      
  # create wavelet 
  srate = 250.0  
  w = calc_w(f,s,M=M)  
  f = 2.0*s*w*srate/M
  #print("morlet freq: ",f)
  bm = signal.morlet(M, w=w, s=s, complete=True)
  if debug: print("raw norm: ",sum(np.abs(bm)), np.linalg.norm(bm))
  if norm:
      bm /= norm
  else:
      bm /= (10.0/s) 
  return bm

def calc_fwhm(x):
  # TODO make dynamic srate
  freqs = np.linspace(0,250,num=len(x))
  fft = np.abs(np.fft.fft(x))
  amax = np.argmax(fft)
  #print("confirm f: ",freqs[amax])
  hmax = fft[amax]/2.0
  fft -= hmax   
  m1,m2 = np.argsort(np.abs(fft))[:2]
  fwhm = abs(freqs[m1] - freqs[m2])
  return fwhm

def test():
  b = make_morlet(12, 2)
  print(calc_fwhm(b))
  plt.plot(b)
  return b   

