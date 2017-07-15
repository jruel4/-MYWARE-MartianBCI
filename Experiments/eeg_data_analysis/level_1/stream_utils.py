# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 06:41:32 2017

@author: MartianMartin
"""

def get_voltage_factor(gain=24.0):
    b2v = 4.5/(2**23 - 1)
    b2v /= gain
    return b2v

def convert_to_volts(x):     
    b2v = get_voltage_factor()
    # Convert to volts
    return x*b2v
