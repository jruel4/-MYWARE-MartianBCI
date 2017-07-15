# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 06:44:01 2017

@author: MartianMartin
"""

import numpy as np
import scipy.signal as sig

# create wavelet 
M = 250 #TODO this must match epoc len, make this dynamic
r = 250.0
s = 1.0/2
w = 9.2*2
f = 2*s*w*r/M
bm = sig.morlet(M, w=w, s=s, complete=True)
bm /= sum(np.abs(bm))

def get_beta_power_from_epochs(epoch_list):

    beta_power = list()
    for epoch in epoch_list:    
        beta_power += [np.abs(np.dot(bm, epoch))]
    
    return np.asarray(beta_power)
    