# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 00:35:03 2017

@author: marzipan
"""

from Block import Block
import numpy as np
from scipy import signal

class block_periodogram (Block):
    
    def __init__(self, _pipe, _NCHAN, _BUFLEN, nfft=None, window=None,remove_dc=False,remove_mirror=False):   
        self.mPipe = _pipe
        self.mnChan = _NCHAN
        self.mBufLen = _BUFLEN
        self.mRemoveDC = remove_dc
        self.mRemoveMirror = remove_mirror

        # Number of fft points
        if nfft is None:
            self.mnFFT = _BUFLEN
        elif nfft < _BUFLEN:
            raise ValueError("nFFT must be equal to or greater than the number of input points")
        else:
            self.mnFFT = nfft
            
        # Assign window
        if window is None:
            self.mWindow = signal.tukey(_BUFLEN, 0.25)
        else:
            assert len(window) == _BUFLEN, "Window and buffer length must be the same size"
            self.mWindow = window
        
        # Input buffer
        self.mBuf = np.zeros([_NCHAN, _BUFLEN])
      
    '''
    Expects buf['default'] to be nparray, shape = (nchan, nsamples)
    '''
    def run(self, _buf, test=False):
        buf = super(block_periodogram, self).get_default(_buf)
        if test:
            print "PER, mnChan: ", self.mnChan
            print "PER, mBufLen: ", self.mBufLen
            print "PER, inbuf: ", buf.shape
        assert buf.shape[0] == self.mnChan, "Input dim-0 must be same as number of channels"
        assert buf.shape[1] <= self.mBufLen, "Input dim-1 must be less than buffer length"

        in_len = buf.shape[1]

        # shift buffer and add new data to the end
        self.mBuf = np.roll(self.mBuf, -(in_len), 1)
        self.mBuf[:,-(in_len):] = buf
        
        # window and fft
        self.mBufWindowed = self.mBuf * self.mWindow
        fft = np.abs(np.fft.fft(self.mBuf, n=self.mnFFT, norm="ortho"))
        
        if self.mRemoveMirror:
            fft = fft[:,:(fft.shape[1])/2]
        if self.mRemoveDC:
            fft = fft[:,1:]
            

        if test:
            print
            print "PER, _buf shape: ", buf.shape
            print "PER, mBuf shape: ", self.mBuf.shape
            print "PER, window shape: ", self.mWindow.shape
            print "PER, windowed buf shape: ", self.mBufWindowed.shape
            print "PER, fft (output) shape: ", fft.shape
            print
        
        return {'default':fft}
        
    def get_output_struct(self):
        return {'default':[self.mnChan,self.mnFFT]}
        
if __name__ == '__main__':
    import pyqtgraph as pg
    from LSLUtils.Base import LSLUtils
    
    utils = LSLUtils()
    
    # input data should be num_chan x num_freq - power values
    nchan = 8
    buflen = 1000
    nfft = 1000

    # Generate and plot fake data
    fake_data = utils.create_sin(length=1000,freqs=[8,12,20], amplitudes=[1,5,2.5],nchan=8)
    fake_data += np.ones_like(fake_data) * 1000 #dc offset
    pg.plot(fake_data[0,:])

    b = block_periodogram(None, nchan, buflen,nfft=nfft)

    # Initial test run
    fft = b.run({'default':fake_data[:,:]},test=True)
    assert fft['default'].shape == (nchan, nfft)
    pqz = pg.plot(np.linspace(0,250,nfft),fft['default'][0,:])
    
#==============================================================================
#     # TEST1: Now remove DC
#     b.mRemoveDC = True
#     fft = b.run({'default':fake_data[:,:]},test=True)
#     assert fft['default'].shape == (nchan, nfft-1), "Remove DC shape test failed"
#     pg.plot(np.linspace(0,250,nfft)[1:], fft['default'][0,:])
#     
#     # TEST2: Now remove half FFT (but not DC)
#     b.mRemoveDC = False 
#     b.mRemoveMirror = True
#     fft = b.run({'default':fake_data[:,:]},test=True)
#     assert fft['default'].shape == (nchan, nfft/2), "Remove mirror shape test failed " + str((nchan,nfft/2)) + " != " + str(fft['default'].shape)
#     pg.plot(np.linspace(0,125,nfft/2), fft['default'][0,:])
#     
#     # TEST3: Now remove half FFT and DC
#     b.mRemoveDC = True
#     b.mRemoveMirror = True
#     fft = b.run({'default':fake_data[:,:]},test=True)
#     assert fft['default'].shape == (nchan, nfft/2 - 1), "Remove mirror shape test failed"
#     pg.plot(np.linspace(0,125,nfft/2)[1:], fft['default'][0,:])    
#==============================================================================
