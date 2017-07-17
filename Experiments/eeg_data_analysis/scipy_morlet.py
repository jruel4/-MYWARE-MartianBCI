# -*- coding: utf-8 -*-
"""
MORLET WAVELET ANALYSIS
"""

import pickle
import scipy.signal as sig
import numpy as np
from matplotlib import pyplot as plt

# load data
streams = pickle.load(open('martin_iso.p','rb'))

'''
We have 4 streams, each with 8 channels of data -> 32 unique series.
Let's design analysis for 1, then apply identical transforms accross all after.
'''

# select first series
snum = 0 # 0-3
chnum = 0 # 0-7
x = streams[snum][:,chnum].copy()

# only look at first half of data
x = x[:len(x)/2]

# The fundamental frequency of this wavelet in Hz is given by:
# f = 2*s*w*r / M where r is the sampling rate.

#TODO calculate versus center frequency f
#TODO calculate FWHM, i.e. frequency smoothing

M = 256
r = 250.0
s = 1.0/2
w = 9.2*2
f = 2*s*w*r/M
print(f)

# create wavelet
bm = sig.morlet(M, w=w, s=s, complete=True)

filtered_signal = sig.filtfilt(bm,[1],x)
power_over_time = np.abs(filtered_signal)

# Create time axis in minutes
t = np.linspace(0, len(x)/250.0/60.0, num=len(x))

# Smooth magnitude signal 
N = int(250*1) # 
std = 80
ma = sig.gaussian(N, std=std, sym=False)
ma /= sum(ma)

# Apply ma filter to power_over_time signal
smooth_power_over_time_N500 = sig.filtfilt(ma, [1], power_over_time)

# plot smoothed data
plt.plot(t, smooth_power_over_time_N500)













