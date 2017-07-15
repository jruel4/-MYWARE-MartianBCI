# -*- coding: utf-8 -*-
"""
MORLET WAVELET ANALYSIS
"""

import pickle
import scipy.signal as sig
import numpy as np
from matplotlib import pyplot as plt

# load data
streams = pickle.load(open('..\\martin_iso.p','rb'))

def get_voltage_factor(gain=24.0):
    b2v = 4.5/(2**23 - 1)
    b2v /= gain
    return b2v
        
b2v = get_voltage_factor()

# Throw out second half of each stream (efficiency sake)
for i in range(len(streams)):
    half_len = int(streams[i].shape[0] / 2.0)
    streams[i] = streams[i][0:half_len,:]

# Convert to volts
for i in range(len(streams)):
    streams[i] *= b2v
    
# remove segments with large amplitude
MAX_AMP = 150e-6 # volts

# Lets just check how many would be cut
# Iterate over data in intervals of 250 and check for p2p > MAX_AMP

num_cut = [[list() for i in range(8)] for j in range(4)]

for snum in range(4):
    for chnum in range(8):
        print("analyzing stream: "+str(snum)+" channel: "+str(chnum))
        # get series
        x = streams[snum][:,chnum]
        # loop through series
        for i in range(len(x)-250-1):
            segment = x[i:i+250]
            if abs(max(segment) - min(segment)) > MAX_AMP:
                num_cut[snum][chnum] += [i]
  
'''              
# Efficient version of above, starts at '#loop through series' comment

# loop through series 
i = 0            
while i < (len(x) - 250): 
    segment = x[i:i+250]
    amax = np.argmax(segment)
    amin = np.argmin(segment)
    furthest_idx = amax if amax > amin else amin
    maxval = segment[amax]
    minval = segment[amin]
    if abs(maxval - minval) > MAX_AMP:
        num_cut[snum][chnum] += [i]
        # skip ahead to furthest
        i = furthest_idx
i += 1
'''  

# num_cut indices are highly redundant => reduce to True/False per epoch loop
# over epochs and check if any of the member indicies of a given epoch are 
# present in the corresponding series' num_cut list

# e.g. num_cut[0][0] = [12, 387, 4000, 40001]
bad_epoch = [[list() for i in range(8)] for j in range(4)]
for snum in range(4):
    for chnum in range(8):
        
        print("reducing stream: "+str(snum)+" channel: "+str(chnum))
        
        bad_idxs = num_cut[snum][chnum]
        bad_set = set(bad_idxs)
        
        # if no bad indexes, move on
        if len(bad_set) < 1:
            continue
        
        # loop over 1 second epochs
        max_idx = int(max(bad_idxs))    
        num_epochs = int(np.ceil(max_idx/250.0))
        bad_epoch[snum][chnum] = [False for i in range(num_epochs)]
        for i in range(num_epochs):
            
            # setup current epoch
            epoch_idxs = list(range(i*250,i*250+250))
            epoch_set = set(epoch_idxs)

            # check for membership in bad set
            if len(epoch_set.intersection(bad_set)) > 0:
                bad_epoch[snum][chnum][i] = True
            

# create wavelet 
M = 256
r = 250.0
s = 1.0/2
w = 9.2*2
f = 2*s*w*r/M
bm = sig.morlet(M, w=w, s=s, complete=True)

#Test wavelet: plt.plot(np.linspace(0,250,len(bm)),np.abs(np.fft.fft(bm)))

for snum in range(4):
    for chnum in range(8):
        
        # copy stream of interest
        x = streams[snum][:,chnum].copy()
        
        # filter
        filtered_signal = sig.filtfilt(bm,[1],x)
        power_over_time = np.abs(filtered_signal)
        
        # Create time axis in minutes
        t = np.linspace(0, len(x)/250.0/60.0, num=len(x))
        
        # plot on new figure
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(t, power_over_time)
        fig.suptitle('Stream: '+str(snum)+', Channel: '+str(chnum), fontsize=20)
        plt.show()








