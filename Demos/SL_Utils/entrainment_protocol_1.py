# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 16:37:34 2017

@author: marzipan
"""

import numpy as np

def protocol_1(state, time):
    #state=[0,0,0,np.random.rand()]
    '''
    Simple biofeedback protocol which: 
        
        1) couples oscillator volume to beta power.
    
        2) sets AM to 0
    
        3) sets FM to cycles of 3 notes 
    
    **NOTE** assuming all frequency power values are scaled 0-1
    
    Output:
        - Returns index of one-hot vector
    
    '''
    AM_OFFSET = 0
    FM_OFFSET = 30
    VOL_OFFSET = 50
    
    beta_idx = 3
    FM_vals = [10, 12, 15]
    FM_cycle_period = 9 # timesteps
    
    # Map beta_power to volume with log scaling
    beta_power = state[beta_idx] + 1e-10 #JCR note: added this to avoid div_by_zero error
    log_vol = np.max([0, (np.log10(beta_power)/2) + 1])
    
    # Map log_vol to VOL_IDX 
    steps = np.linspace(0,1,num=10)
    diffs = np.abs(steps - log_vol)
    VOL_IDX = np.argmin(diffs)
    
    # Calculate FM cycle state
    FM_state = int(np.floor(time/FM_cycle_period))%3 # e.g. 000,111,222,000,111,...
    FM_IDX = FM_vals[FM_state]
    
    # Set AM to const
    AM_IDX = 0

    # Determine output type by state
    if time % 3 == 0:        # AM
        OUT =  AM_IDX + AM_OFFSET
    elif (time+1) % 3 == 0:  # FM
        OUT = FM_IDX + FM_OFFSET
    else:                    # VOL
        OUT = VOL_IDX + VOL_OFFSET
    
    arr = np.zeros(60)
    arr[OUT] = 1
    return arr

def protocol_1_generator(time_list):
    inputs = []
    outputs= []
    for t in time_list:
        state = [0,0,0,np.random.rand()]
        action_idx = protocol_1(state, t)
        inputs += [state]
        outputs += [action_idx]
    # Convert to format?
    return {'a':inputs, 'b': outputs}









