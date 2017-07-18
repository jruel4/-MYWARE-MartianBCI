#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 23:34:52 2017

@author: marzipan
"""

'''

Automatic graph scaling

Insight into magnitudes

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
    
    def __init__(self, _pipe, _num_freqs):  
        '''
        Log normalize against moving average.
        '''
        self.NUM_FREQS = _num_freqs
        self.NUM_CHAN = 8
        self.BASE_LINE_PERIOD_DURATION = 10 * 250 # duration x time srate
        self.mPipe = _pipe
        self.buf = deque([np.zeros((self.NUM_CHAN, self.NUM_FREQS)) for i in range(self.BASE_LINE_PERIOD_DURATION)], self.BASE_LINE_PERIOD_DURATION)
        
    def run(self, _buf):
        
        inbuf = _buf['default']
        assert inbuf.shape == (self.NUM_CHAN, self.NUM_FREQS)

        # add inbuf to loca buf
        self.buf.append(inbuf)
        
        # convert to 3d array
        d = np.asarray(self.buf)
        assert d.shape == (self.BASE_LINE_PERIOD_DURATION, self.NUM_CHAN, self.NUM_FREQS), str(d.shape)
        
        #TODO get average accross time for each frequency and electrode
        avg = np.mean(d, axis=0)
        
        assert inbuf.shape == avg.shape
        
        decibel_power = 10 * np.log10(inbuf / avg)
        
        return {'default':decibel_power}
      
    def get_output_struct(self):
        return {'default' : (self.NUM_CHAN,self.num_freqs)}
    

if __name__ == '__main__':
    
    # input data should be num_chan x num_freq - power values
    num_chan = 8
    num_freq = 20
    fake_data = np.arange(num_chan*num_freq).reshape((num_chan, num_freq))
    
    b = block_normalize_periodogram(None, num_freq)
    r = b.run({'default':fake_data})
    assert r['default'].shape == (num_chan, num_freq)
    print(r)
    
    
    
    
    
    
    
    
    
    
    