# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 00:17:44 2017

@author: marzipan
"""

import numpy as np

'''
Policy Network Spec

Input:
    - eeg state
    - previous action
    - time
    
Output:
    - Current Action
        - Set beat frequency (AM)     [1-30 discretized] (delta-gamma)
        - Set base frequency (FM)     [1-20 discretized] (notes in key)
        - Set amplitude      (Volume) [1-10 discretized] (decibel volume)
        
NOTE** must learn to output in cycles of 3
'''

# state = [delta, theta, alpha, beta] (powers)

def protocol_mod_4(time):
    arr = np.zeros(4)
    arr[time%4] = 1
    return arr

def create_protocol(n):
    def protocol_n(time):
        arr = np.zeros(n)
        arr[time%n] = 1
        return arr
    return protocol_n

def protocol_0(time):
    state=[0,0,0,0]
    action=None
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

# Gets passed in an array! Generator
def protocol_1_follow_beta(time_array):
    state = [0]*50
    action = None
    '''
    Simple biofeedback protocol which: 
        
        1) couples oscillator volume to beta power.
    
        2) sets AM to 0 / if AM
    
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
    
    if not hasattr(time_array, '__iter__'):
        raise TypeError("protocol_1_follow_beta needs to be passed an iterable")
    
    # Generate beta states
    audio_states = np.zeros([len(time_array), 60])
    eeg_states = np.zeros([len(time_array), 50])
    for idx in range(len(eeg_states)): eeg_states[idx][beta_idx] = (idx % 100) / 100
    for idx in range(len(audio_states)):
        # Amplitude Modulation Step
        if idx % 3 == 0:
            OUT = 0 + AM_OFFSET
        # Base Frequency Step
        elif (idx+1) % 3 == 0:
            FM_IDX = FM_vals[idx % 3]
            OUT = FM_IDX + FM_OFFSET
        # Base Volume Step
        elif (idx+2) % 3 == 0:
            beta_power = np.max(eeg_states[idx]) + 1e-10 #JCR note: added this to avoid div_by_zero error
            log_vol = np.max([0, (np.log10(beta_power)/2) + 1])
            steps = np.linspace(0,1,num=10)
            diffs = np.abs(steps - log_vol)
            VOL_IDX = np.argmin(diffs)            
            OUT = VOL_IDX + VOL_OFFSET
        audio_states[idx][OUT] = 1

    # I don't know if this is necessary - should audio states be 1 step ahead of EEG?
#    audio_states = np.roll(audio_states, 1, axis=0)
    out_dict = {'a': np.concatenate([audio_states,eeg_states],axis=1).tolist(),
                'b': audio_states.tolist()}
    return out_dict

# --1-- easy model

# Start with less noisy data generation
# i.e. hold state constant, hold action constant, input only time
# Can it directly classify time with random data?

#NUM_SAMPLES = int(1E5)
#samples = np.asarray([protocol_0(i,[0,0,0,0.5], 0) for i in range(NUM_SAMPLES)])



# 

# --2-- medium model

# Vary beta power cyclically

# --3-- hard model

# Vary beta power cyclically and other state params randomly






