# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 05:07:32 2017

@author: MartianMartin
"""

import mne
import pickle

load_streams
streams = pickle.load(open('martin_iso.p','rb'))

ch_names = 8
sfreq = 250
ch_types = ['eeg' for i in range(8)]

# create raw array from import data
mne_streams = []
for s in streams:
    mne_streams += [mne.EvokedArray(s.transpose(), mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types))]
    
    
raw = mne_streams[0]



scalings = 'auto'  # Could also pass a dictionary with some value == 'auto'
raw.plot(n_channels=1, scalings=scalings, title='Auto-scaled Data from arrays',
         show=True, block=True)
         
raw.plot_psd()

eog_events = mne.preprocessing.find_eog_events(raw)

