# -*- coding: utf-8 -*-
"""
B-Power MORLET WAVELET ANALYSIS
"""

# External Imports
import pickle

# Internal Imports
from stream_utils import convert_to_volts, v2uv
from get_clean_epochs import get_clean_epochs
from tf_utils import get_beta_power_from_epochs
from plot_utils import plot_new_figure

# load data
streams = pickle.load(open('..\\martin_iso.p','rb'))

CHANS = (0,8)
STREAMZ = (3,4)

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
        plot_new_figure(d['timestamps_minutes'], v2uv(beta_power), snum, chnum)
        









