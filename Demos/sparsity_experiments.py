# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 19:07:24 2017

@author: MartianMartin
"""

from matplotlib import pyplot as plt
import numpy as np

def make_p_sparse(v, p):
    p = 1.0 - p
    fft = np.fft.fft(v)
    thresh = np.sort(np.abs(fft))[int(len(v)*p)]
    for i in range(len(fft)):
        if np.abs(fft[i]) < thresh:
            fft[i] = 0
    return np.real(np.fft.ifft(fft)).astype(np.int16)

def plt_sparse(v, p=.1):
    plt.plot(make_p_sparse(v,p))
    
    
from pydub import AudioSegment
sound = AudioSegment.from_mp3("/path/to/file.mp3")
sound.export("/output/path", format="wav")    
    
amax = 29000
amin = -29000

def make_sine(freq, srate, duration, phase):
    time_steps = srate * duration
    return np.sin([2*np.pi*freq*i/time_steps+phase for i in range(int(time_steps))])

sines = make_sine(400, rate, 10) * .001
idx = 0
phase = 0
for f in range(400-15,400+49, 7):
    sines += make_sine(f, rate, 10, np.pi/phase)*gauss[idx]
    phase += 1
    idx += 6
    
sines *= 58000/(sines.max() - sines.min())
