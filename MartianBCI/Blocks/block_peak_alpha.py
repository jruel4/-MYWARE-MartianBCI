#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 22:25:32 2017

@author: marzipan
"""

if __name__ == "__main__":
    from Block import Block
else:
    from .Block import Block
import numpy as np
from MartianBCI.Utils.wavelets import make_morlet

class block_peak_alpha (Block):
    '''
      Finding useres IAF (peak alpha) means performing an argmax on frequency power
      data from a known range ~[7-14]. 
      
      block should thus return both the IAF frequency value and the power value at that frequency.
  
    '''  
    def __init__(self, _pipe):  
        '''
        f_fwhm is a list of tuples where each tuple is (frequency, full-width-half-max)
        '''
        self.mPipe = _pipe
        
        self.freqs = np.arange(7,14,0.1)
        self.NUM_CHAN = 8
        self.EPOCH_LEN = 250
        # create wavelets 
        self.wavelets_mat = np.asarray([make_morlet(f,2,M=self.EPOCH_LEN) for f in self.freqs]).transpose()
        self.num_freqs = len(self.freqs)
        
        assert self.wavelets_mat.shape == (self.EPOCH_LEN, self.num_freqs)
        
    def run(self, _buf):
        
        inbuf = _buf['data']
        valid = _buf['valid']

        # check input size
        assert inbuf.shape == (self.NUM_CHAN, self.EPOCH_LEN)        
        power = np.abs(np.dot(inbuf, self.wavelets_mat)) # Volts
        assert power.shape == (self.NUM_CHAN, self.num_freqs)        
        amax = np.argmax(power, axis=1)
        assert amax.shape == (self.NUM_CHAN,)
        assert valid.shape == (self.NUM_CHAN,)
        
        # get corresponding freq, power for determined argmax
        iaf = self.freqs[amax]
        iafp = power[np.arange(self.NUM_CHAN),amax]
        
        # Negate invalid channels
        iafp *= np.asarray([1 if i else -1 for i in valid])
        
        # Output is (nchan, nfreqs)
        return {'peak_alpha_freq':iaf, 'peak_alpha_power':iafp}
      
    def get_output_struct(self):
        return {'peak_alpha_freq' : (self.NUM_CHAN,1),
                'peak_alpha_power': (self.NUM_CHAN, 1)}
        
    def test(self):
        fake_data = np.asarray([np.sin(2*np.pi*(8+i) * np.linspace(0,1,250)) for i in range(8)])
        fake_valid = np.asarray([False if not i else True for i in range(8)])
        d = {'data':fake_data,
             'valid':fake_valid}  
        r = self.run(d)
        return r

if __name__ == '__main__':
    d = block_peak_alpha(None)
    r = d.test()      
    print(r)
      
      
      
      
      
      
      
      
    
    
    
    
    
    
    
    
    
    
      
      
      
      