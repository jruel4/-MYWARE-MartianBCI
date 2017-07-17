# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 23:32:28 2017

@author: marzipan
"""

from scipy import signal
from .Block import Block
from Queue import deque
import numpy as np

class block_fir (Block):
    
    def __init__(self, _pipe, _CHANNELS):   
        self.mPipe = _pipe
        
        self.n_taps = 250
        self.fir_coeffs = signal.firwin(self.n_taps, [16.0, 20.0], pass_zero=False, nyq=125.0)
        self.buf = deque([np.zeros([_CHANNELS]) for i in range(self.n_taps)], maxlen=self.n_taps)
        self.once = True
        
        self.mBuf = np.zeros([_CHANNELS,self.n_taps])
        
    def run(self, inbuf):
        
        for i in range(len(inbuf)):
            #remove the oldest value from the buffer
            self.mBuf = np.delete(self.mBuf,0,1)
            #and insert the next (& format input to be #chan x #samples)
            _buf_in = np.asarray([[x] for x in inbuf.pop()])
            self.mBuf = np.append(self.mBuf,_buf_in,1)        
        
#==============================================================================
#         for i in range(len(inbuf)):
#           if self.once:
#               a=inbuf.pop()
#               print a
#               self.buf.append([[x] for x in a])
#           else:
#               self.buf.append([[x] for x in inbuf.pop()])
#           print "\n\n\n"
#==============================================================================
        if self.once:
            print np.dot(self.fir_coeffs, np.transpose(self.mBuf)).shape
            print np.asarray(self.mBuf).shape
        self.once = False
        return {'filtered':np.dot(self.fir_coeffs, np.transpose(self.mBuf))}
#                'raw':np.asarray(np.transpose(self.mBuf))[-1,:]}
    
    def get_output_struct(self):
        return {'filtered':8}
      
    def set_filter(self, f1, f2):
      self.fir_coeffs = signal.firwin(self.n_taps, [f1, f2], pass_zero=False, nyq=125.0)
    
    
    
    
    
    
    