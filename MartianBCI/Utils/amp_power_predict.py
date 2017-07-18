#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 01:02:34 2017

@author: marzipan
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sig

def mk_sine(f=12,amp=1,duration=1,srate=250):
    return amp*np.sin(2*np.pi*f * np.linspace(0,duration,duration*srate))


sine = mk_sine()

freqs = np.linspace(0,250,250)
fft = np.abs(np.fft.fft(sine, norm="ortho"))


plt.plot(fft)
print("max freq: ", freqs[np.argmax(fft)])
print("max power: ", fft.max())

# np.fft.fft in unscaled by default, ortho scales by 1/sqrt(N)... does this preserve norm?

# first try unscaled...

print("p2p amp before: ",sine.max() - sine.min())
fft = np.fft.fft(sine, norm="ortho")
sine_after = np.fft.ifft(fft)
print("p2p after: ", sine_after.max() - sine_after.min())

# Ok, so norm presevation requires 1/sqrt scaling? lets check full norm

print("p2p amp before: ",np.linalg.norm(sine))
fft = np.fft.fft(sine, norm="ortho")
print("norm fft: ",np.linalg.norm(fft))
sine_after = np.fft.ifft(fft)

print("p2p after: ", np.linalg.norm(sine_after))

from wavelets import make_morlet, calc_fwhm

# TODO try scaling s...
# TODO try norming wavelet

f = 12
s = .9
norm = 1.0
for s in np.linspace(.5,1.2,num=4):
    wavelet = make_morlet(f, s=s)
    print("fwhm: ",calc_fwhm(wavelet))
    
    nb4 = np.linalg.norm(sine)
    p2pa = sine.max()-sine.min()
    power = np.abs(np.dot(sine, wavelet))
    
    print("norm before: ",np.linalg.norm(sine))
    print("p2p amp before: ",sine.max() - sine.min())
    print("wavelet power: ",np.abs(np.dot(sine, wavelet)))
    print()
    print("norm/power = ",nb4/power)
    print("p2p/power  = ",p2pa/power) # want this to be ~1
    print()
    print()
    
# seems like more spread i.e. higher fwhm == more power at the same freq?

# Ok, what factor do we normalize by to get something like 1 volts power for 1 volts amplitude?








