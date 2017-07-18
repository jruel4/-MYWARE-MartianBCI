#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 19:31:23 2017

@author: marzipan
"""

from scipy import signal
import numpy as np


fwhm_dict = {2:.6,
             4:1.0,
             6:1.2}


def calc_w(f, s,M=250.0):
  srate = 250.0
  # f = 2*s*w*srate/M
  # f/2/srate*M = s*w
  w = f/2.0/srate*M/s
  return w

def make_morlet(f,fwhm=4,srate=250.0,M=250.0):
  '''
  fwhm = full width half max in hz, must be 2, 4 or 6.
  '''
  s = fwhm_dict[fwhm]
  # create wavelet 
  srate = 250.0  
  w = calc_w(f,s,M=M)  
  f = 2.0*s*w*srate/M
  #print("morlet freq: ",f)
  bm = signal.morlet(M, w=w, s=s, complete=True)
  bm /= sum(np.abs(bm))    
  return bm

def calc_fwhm(x):
  # TODO make dynamic srate
  freqs = np.linspace(0,250,num=len(x))
  fft = np.abs(np.fft.fft(x))
  amax = np.argmax(fft)
  #print("confirm f: ",freqs[amax])
  hmax = fft[amax]/2.0
  fft -= hmax   
  amin = np.argmin(abs(fft))  
  fwhm = abs(freqs[amax] - freqs[amin])*2  
  return fwhm
  



