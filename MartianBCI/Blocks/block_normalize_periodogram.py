#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 23:34:52 2017

@author: marzipan
"""

'''

Make test data with proper untits / scaling

Make lsl recorder in our gui?

'''

if __name__ == "__main__":
    from Block import Block
else:
    from .Block import Block
import numpy as np
from collections import deque

class block_normalize_periodogram (Block):
    
    def __init__(self, _pipe, _num_freqs, _num_chan=8, continuous_baseline=False):  
        '''
        Log normalize against moving average.
        '''
        self.srate = 250.0
        self.continuous_baseline = continuous_baseline
        self.baseline_sample_count = 0
        self.NUM_FREQS = _num_freqs
        self.NUM_CHAN = _num_chan
        self.BASE_LINE_PERIOD_DURATION = int(10 * self.srate) # duration x time srate
        self.mPipe = _pipe
        self.buf = deque([np.zeros((self.NUM_CHAN, self.NUM_FREQS)) for i in range(self.BASE_LINE_PERIOD_DURATION)], self.BASE_LINE_PERIOD_DURATION)
        
    def run(self, _buf,debug=False):
        
        inbuf = _buf['default']
        assert inbuf.shape == (self.NUM_CHAN, self.NUM_FREQS)

        # add inbuf to local buf 
        if self.continuous_baseline or (self.baseline_sample_count < self.BASE_LINE_PERIOD_DURATION):
            self.baseline_sample_count += 1
            self.buf.append(inbuf)
            
            if not self.continuous_baseline and (self.baseline_sample_count == self.BASE_LINE_PERIOD_DURATION):
                print ("NOTE: Baseline recording completed.")
        
        # convert to 3d array
        d = np.asarray(self.buf)
        assert d.shape == (self.BASE_LINE_PERIOD_DURATION, self.NUM_CHAN, self.NUM_FREQS), str(d.shape)
        
        # If power vals are zero, exclude them from average
        
        # First we need to count the number of zeros in each epoch
        negatives = d < 0
        negatives_sum = np.sum(negatives, axis=0)
        Ns= (-1*negatives_sum) + self.BASE_LINE_PERIOD_DURATION
        
        # Now set negatives to zero by multiplying by zero or one
        one_or_zero = (d >= 0) * 1.0
        zerod_data = d * one_or_zero
        
        # Now calculate mean by summing and dividing by len - negative sum
        summed_data = np.sum(zerod_data, axis=0)
        avg = summed_data / Ns
        
        if not self.continuous_baseline and (self.baseline_sample_count == self.BASE_LINE_PERIOD_DURATION):
            print ( "Baseline avg: ", avg)
            self.baseline_sample_count += 1
        
        assert inbuf.shape == avg.shape
        
        # Check for div by 0 error here
        if not avg or avg == 0.0:
            divisive = inbuf
        else:
            divisive = inbuf / avg

#        decibel_power = 10 * np.log10(inbuf / avg)

        return {'default':divisive}
#        return {'default':decibel_power}
      
    def get_output_struct(self):
        return {'default' : (self.NUM_CHAN,self.num_freqs)}
    
    def record_baseline(self, duration_seconds=10):
        self.BASE_LINE_PERIOD_DURATION = int(duration_seconds * self.srate)
        self.baseline_sample_count = 0
        print("NOTE: Recording baseline...")
        

if __name__ == '__main__':
    
    # input data should be num_chan x num_freq - power values
    num_chan = 2
    num_freq = 2
    
    fake_data = np.asarray([[10e-6, 10e-6],[10e-6, 10e-6]])
    b = block_normalize_periodogram(None, num_freq, _num_chan=num_chan)
    # fill up buffer
    for i in range(b.BASE_LINE_PERIOD_DURATION): b.run({'default':fake_data})
    fake_data = np.asarray([[11e-6, 12e-6],[13e-6, 30e-6]])
    r = b.run({'default':fake_data},debug=True)
    
    assert r['default'].shape == (num_chan, num_freq)
    print(r)
    
    
    
    
    
    
    
    
    
    
    