#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 05:32:34 2017

@author: marzipan
"""

if __name__ == "__main__":
    from Block import Block
else:
    from .Block import Block
import numpy as np
from collections import deque

class block_noise_reject (Block):
    
    def __init__(self, _pipe):   
        self.mPipe = _pipe
        
        self.EPOCH_LEN = 250
        self.NUM_CHAN = 8
        self.NUM_SAMPLES_IN = 1
        self.NOISE_THRESH = 150e-6 # Volts
        self.B2V = 4.5/(2**23-1)/24.0
        
        self._buf = deque([np.zeros(self.NUM_CHAN) for i in range(self.EPOCH_LEN)], maxlen=self.EPOCH_LEN)
        
    def run(self, _buf):
      inbuf = super(block_noise_reject, self).get_default(_buf)

      #TODO add smart check to ensure input is not already converted to Volts
      
      # Check input dimensions
      assert inbuf.shape == (self.NUM_CHAN, self.NUM_SAMPLES_IN)
      
      # Split samples to add to buff
      list_of_samples = inbuf.transpose().tolist()
      self._buf.extend(list_of_samples)
      
      # Convert deque to array
      d = np.asarray(self._buf).transpose() # num_chan x num_samples
      
      # convert to voltnum
      d *= self.B2V
      
      # get min, max of each column to see if artifact
      mins = d.min(axis=1)
      maxs = d.max(axis=1)
      
      valid = (maxs - mins) < self.NOISE_THRESH
      
      return {'valid': valid,
              'data': d}
      
    def get_output_struct(self):
        return {'valid':(self.NUM_CHAN),
                'data':(self.NUM_CHAN, self.EPOCH_LEN)}
      

    
a = block_noise_reject(None)
fd = np.transpose([[1,1,1,1,1,1,1,1e15]])

r = a.run({'default':fd})
    