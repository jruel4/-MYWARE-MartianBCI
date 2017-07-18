#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 02:21:39 2017

@author: marzipan
"""
from amp_power_predict import mk_sine
from wavelets import make_morlet
import numpy as np

def generate_sine_amp(uV):
    b2v = 2.4/(2**23-1)/24.0    
    scale = uV/1e6/b2v
    sine_amp = scale/2
    return sine_amp

def b2v(x):
    b2v = 2.4/(2**23-1)/24.0   
    return x*b2v

if __name__ == "__main__":
    
    f = 12
    
    a = generate_sine_amp(10)
    
    sine = mk_sine(f=f, amp=a) # IN BITS
    
    sine_volts = b2v(sine)
    
    power_volts = np.abs(np.dot(sine_volts, make_morlet(f)))
    
    power_uV = power_volts * 1e6
    
    print("Power uV: ",power_uV)
    




