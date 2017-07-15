# -*- coding: utf-8 -*-
"""
MORLET WAVELET ANALYSIS
"""

import pickle
import scipy.signal as sig
import numpy as np
from matplotlib import pyplot as plt

from stream_utils import convert_to_volts
from get_clean_epochs import get_clean_epochs
from tf_utils import get_beta_power_from_epochs
from plot_utils import plot_new_figure

#TODO use smaller noise threshold for B-power

# load data
streams = pickle.load(open('..\\martin_iso.p','rb'))

CHANS = (0,8)
STREAMZ = (0,1)

for snum in range(*STREAMZ):
    for chnum in range(*CHANS):
        
        print("Proccessing Stream: "+str(snum)+", Channel: "+str(chnum))
        
        # select series
        x = streams[snum][:,chnum]
        
        # convert to volts
        x = convert_to_volts(x)
        
        # get clean epochs
        d = get_clean_epochs(x)

        # get beta power for each epoch
        beta_power = get_beta_power_from_epochs(d['clean_epochs'])
        
        # plot figure
        plot_new_figure(d['timestamps_minutes'], beta_power, snum, chnum)
        
        
        









