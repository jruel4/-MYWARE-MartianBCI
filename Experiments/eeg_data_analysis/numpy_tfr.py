# -*- coding: utf-8 -*-
"""
Filter-Hilbert Method
"""

import pickle
import scipy.signal as sig
import numpy as np

# load data
streams = pickle.load(open('martin_iso.p','rb'))

'''
We have 4 streams, each with 8 channels of data -> 32 unique series.
Let's design analysis for 1, then apply identical transforms accross all after.

Note: morlet
scipy.signal.morlet(M, w=5.0, s=1.0, complete=True)
'''

# select first series
snum = 0 # 0-3
chnum = 0 # 0-7
x = streams[snum][:,chnum]

#TODO convert units to microvolts

# create filter
b = sig.firwin(1024, [16.0, 20.0], pass_zero=False, nyq=125.0)

# Wavelet or bandpass filter? - bandpass
filtered_signal = sig.filtfilt(b,[1],x)
analytic_signal = sig.hilbert(filtered_signal) # why so slow? huge fft...
power_over_time = np.abs(analytic_signal)

# Create time axis in minutes
t = np.linspace(0, len(x)/250.0/60.0, num=len(x))

# Smooth magnitude signal 
N = 500 # half a second average
ma = np.asarray([1.0/N for i in range(N)])

# Apply ma filter to power_over_time signal
smooth_power_over_time_N500 = sig.filtfilt(ma, [1], power_over_time)

# Plot
plt.plot(t, smooth_power_over_time_N500)









