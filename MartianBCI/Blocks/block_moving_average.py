# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 12:24:26 2017

@author: marzipan
"""

from Block import Block
import numpy as np
from scipy import signal

class block_moving_average (Block):
    
    def __init__(self, _pipe, _NCHAN, _MA_LEN, _NFREQS = None, remove_negatives = True):
        self.mPipe = _pipe
        self.mnChan = _NCHAN
        self.mMALen = _MA_LEN
        
        # If _NFREQS is provided assume we're operating on freq data and only output (nchan, nfreqs)
        # otherwise, output (nchan, nsamples)
        # this is shitty / hacky, TODO: Fix / standardize
        self.mnFreqs = _NFREQS

        if _NFREQS == None:
            self.isTS = True
        else:
            self.isTS = False

        self.mRemoveNegatives = remove_negatives

        # Input buffer
        if self.isTS:
            self.mBuf = np.zeros([_NCHAN, _MA_LEN - 1]) # ts
        else:
            self.mBuf = np.zeros([_NCHAN * _NFREQS, _MA_LEN - 1]) #freq
      
    '''
#    Expects buf['default'] to be nparray, shape = (nchan, nfreqs, nsamples)
    '''
    def run(self, _buf, test=False):
        buf = super(block_moving_average, self).get_default(_buf)
        in_len = buf.shape[1]

        assert buf.shape[0] == self.mnChan, "Input dim-0 must be same as number of channels, bufshape is: " + str(buf.shape)
        if not self.isTS:
            assert buf.shape[1] == self.mnFreqs, "Input dim-1 must be same as number of freqs, bufshape is: " + str(buf.shape)
        
        if not self.isTS:
            buf = buf.reshape(-1,1)
            in_len = 1

        buf_tmp = np.concatenate([self.mBuf, buf],axis=1)

        # ma_mask is used both the zero ma_buffer (where appropriate) and to calculate the number of valid points per average
        ma_mask = (~np.isnan(buf_tmp)).astype(int) # remove nans from average calculation
        if self.mRemoveNegatives:
            ma_mask *= (buf_tmp >= 0).astype(int) # remove negatives from average calculation
        
        # remove nans so that everything isn't fucked
        buf_cleaned = np.nan_to_num(buf_tmp)
        buf_cleaned = buf_cleaned * ma_mask # clean up negatives 

        ma_kernel = np.ones((1,self.mMALen))
        buf_smoothed = signal.convolve2d(buf_cleaned, ma_kernel,mode='valid',fillvalue=1)
        n_samples_per_point = signal.convolve2d(ma_mask, ma_kernel,mode='valid',fillvalue=1)
        
        # make sure we're not trying to divide by zero for any points that were completely invalid
        invalids = (n_samples_per_point == 0)
        n_samples_per_point += invalids.astype(int)
        out = buf_smoothed / n_samples_per_point
        
        # Make any invalids nan so that pyqtgraph can interpolate them
        out[invalids] = np.nan

        # shift buffer and add new data to the end
        replace_len = min([in_len, self.mMALen-1]) #self.mMALen - 1 is used here because the newest samples will, at mininum, make up 1 point of the new moving average
        self.mBuf = np.roll(self.mBuf, -(replace_len), 1)
        self.mBuf[:,-(replace_len):] = buf[:,-(replace_len):]

        dropped = self.mMALen - (n_samples_per_point - invalids.astype(int))

        if test:
            print
            print "MA, _buf shape: ", buf.shape
            print "MA, mBuf shape: ", self.mBuf.shape
            print "MA, buf smoothed shape: ", buf_smoothed.shape
            print "MA, out shape: ", out.shape
            print "MA, samples per point: ", self.mMALen - (n_samples_per_point - invalids.astype(int))
            print
            
        if not self.isTS:
            out = out.reshape((self.mnChan,self.mnFreqs))
        return {'default':out,'dropped':dropped}
        
    def get_output_struct(self):
        return {'default':[self.mnChan,-1], 'dropped':[self.mnChan,-1]}




if __name__ == '__main__':

    import pyqtgraph as pg 
    from Utils.lsl_utils import create_sin

    win = pg.GraphicsWindow(title="Plot auto-range examples")
    win.resize(800,600)
    p1 = win.addPlot()
    
    # input data should be num_chan x num_freq - power values
    nchan = 8
    buflen = 1000
    ma_len = 50

    # Generate and plot fake data
    fake_data = create_sin(length=buflen,freqs=[5], amplitudes=[1],nchan=8)
    fake_data[:,50:150] = -2
    fake_data += 1
    p1.plot(fake_data[0,:],title="Fake data, raw, 5Hz")

    b = block_moving_average(None, nchan, ma_len,remove_negatives=False)

    # first fill ma buf
    _ = b.run({'default':fake_data[:,:]},test=False)

    # Initial test run, keep negatives
    p2 = win.addPlot()

    ma = b.run({'default':fake_data[:,:]},test=True)
    assert ma['default'].shape == (nchan, buflen)
    p2.plot(ma['default'][0,:],title="Not removing negatives",connect='finite')
    p2.plot(ma['dropped'][0,:],color='orange')

    b.mRemoveNegatives = True    
    
    # Initial test run
    p3 = win.addPlot()
    
    ma = b.run({'default':fake_data[:,:]},test=True)
    assert ma['default'].shape == (nchan, buflen)
    p3.plot(ma['default'][0,:],title="Removing negatives, PyQtGraph interpolates",connect='finite')
    p3.plot(ma['dropped'][0,:],pen=pg.intColor(1, hues=8))    