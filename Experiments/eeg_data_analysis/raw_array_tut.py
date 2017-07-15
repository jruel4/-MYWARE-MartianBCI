# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 05:27:13 2017

@author: MartianMartin
"""

import numpy as np
#import neo
import mne

sfreq = 1000  # Sampling frequency
times = np.arange(0, 10, 0.001)  # Use 10000 samples (10s)

sin = np.sin(times * 10)  # Multiplied by 10 for shorter cycles
cos = np.cos(times * 10)
sinX2 = sin * 2
cosX2 = cos * 2

# Numpy array of size 4 X 10000.
data = np.array([sin, cos, sinX2, cosX2])

# Definition of channel types and names.
ch_types = ['mag', 'mag', 'grad', 'grad']
ch_names = ['sin', 'cos', 'sinX2', 'cosX2']

# It is also possible to use info from another raw object.
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

raw = mne.io.RawArray(data, info)

# Scaling of the figure.
# For actual EEG/MEG data different scaling factors should be used.
scalings = {'mag': 2, 'grad': 2}

raw.plot(n_channels=4, scalings=scalings, title='Data from arrays',
         show=True, block=True)

# It is also possible to auto-compute scalings
scalings = 'auto'  # Could also pass a dictionary with some value == 'auto'
raw.plot(n_channels=1, scalings=scalings, title='Auto-scaled Data from arrays',
         show=True, block=True)
         
         
         
         
         
         
# Read epochs
picks = mne.pick_types(raw.info, meg=False, eeg=False, stim=False, eog=False,
                       include=['0'], exclude='bads')
tmin, tmax = -0.1, 0.1
epochs = mne.Epochs(raw, ecg_events, 999, tmin, tmax, picks=picks,
                    proj=False)
data = epochs.get_data()

print("Number of detected ECG artifacts : %d" % len(data))









         