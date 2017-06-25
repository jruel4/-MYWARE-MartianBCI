# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 12:44:46 2017

@author: marzipan
"""

import numpy as np

def autocorrelation(v,d):
    return np.dot(v, get_n_del_vec(v,d))

def get_n_del_vec(v,d):
    return np.concatenate((v[d:], v[:d]))


