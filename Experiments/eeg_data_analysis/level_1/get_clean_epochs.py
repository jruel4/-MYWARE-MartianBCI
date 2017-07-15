import scipy.signal as sig
import numpy as np
from matplotlib import pyplot as plt
import sys

def get_clean_epochs(x):
    
    # remove segments with large amplitude
    MAX_AMP = 150e-6 # volts
    
    EPOCH_LEN = 250
     
    # Lets just check how many would be cut
    # Iterate over data in intervals of EPOCH_LEN and check for p2p > MAX_AMP
    num_cut = list()
    
    # loop through series 
    i = 0            
    while i < (len(x) - EPOCH_LEN): 
        segment = x[i:i+EPOCH_LEN]
        amax = np.argmax(segment)
        amin = np.argmin(segment)
        furthest_idx = amax if amax > amin else amin
        furthest_idx += i # make it relative to whole series
        maxval = segment[amax]
        minval = segment[amin]
        if abs(maxval - minval) > MAX_AMP:
            num_cut += [i+amin, i+amax]
            # skip ahead to furthest
            i = furthest_idx
        i += 1 
        
    # sort indexes to cut
    num_cut = sorted(num_cut)
    bad_set = set(num_cut)
    
    # if no bad indexes, move on
    if len(bad_set) < 1:
        print("no artifacts in this series")
        sys.exit()
    
    # loop over 1 second epochs
    max_idx = int(max(bad_set))    
    num_epochs = int(np.ceil(max_idx/EPOCH_LEN)) + 1
    # make sure num epochs not more than available in series
    max_epochs = int(np.floor(len(x)/EPOCH_LEN))
    num_epochs = num_epochs if num_epochs < max_epochs else max_epochs
    
    bad_epoch = [False for i in range(num_epochs)]
    for i in range(num_epochs):
        
        # setup current epoch
        epoch_idxs = list(range(i*EPOCH_LEN,i*EPOCH_LEN+EPOCH_LEN))
        epoch_set = set(epoch_idxs)
    
        # check for membership in bad set
        if len(epoch_set.intersection(bad_set)) > 0:
            bad_epoch[i] = True
            
    '''
    We can do this in one loop now! All code below should be eliminated
    and worked into look above this comment.
    '''
    
    # Create 32 new series, with each series a concatenation of clean epochs only
    clean_epochs = []
    clean_epoch_idxs = []
            
    series = x
    # loop over epochs
    num_epochs = max_epochs
    for i in range(num_epochs):
        
        if i > (len(bad_epoch)-1):
            # epoch is good
            epoch = series[i*EPOCH_LEN:i*EPOCH_LEN+EPOCH_LEN]
            clean_epochs += [np.asarray(epoch)]
            clean_epoch_idxs += [i*EPOCH_LEN]
            continue
        else:
            if not bad_epoch[i]:
                # epoch is good
                epoch = series[i*EPOCH_LEN:i*EPOCH_LEN+EPOCH_LEN]
                clean_epochs += [np.asarray(epoch)]
                clean_epoch_idxs += [i*EPOCH_LEN]
            else:
                # epoch is bad
                pass
    
    # convert clean_epoch_idxs to timestamps (minutes)
    timestamps = [t/250.0/60.0 for t in clean_epoch_idxs]
    
    return {'clean_epochs' : clean_epochs, 'timestamps_minutes': timestamps}
    
    
    
    
    
    
    
    
    
