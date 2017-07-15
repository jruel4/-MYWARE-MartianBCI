# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 03:57:53 2017

@author: MartianMartin
"""

from matplotlib import pyplot as plt
import pickle

streams = pickle.load(open('martin_iso.p','rb'))

def plot_data(streams, sidx, chan, begin_t, end_t):
    SRATE=250
    stream = streams[sidx]
    series = stream[:,chan]
    bidx = int(begin_t * SRATE * 60.0)
    eidx = int(end_t * SRATE * 60.0)
    segment = series[bidx:eidx]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(segment)
    plt.show()
    
    
def generate_segments():
    pass