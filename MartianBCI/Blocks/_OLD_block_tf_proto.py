# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 08:09:51 2017

@author: marzipan
"""

import tensorflow as tf

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)

update = tf.assign(W, W*2)

linear_model = W * x + b

init = tf.global_variables_initializer()

#==============================================================================
# x = tf.placeholder(tf.complex64)
# 
# _a = [1 for i in range(256)]
# _b = [2 for i in range(256)]
# 
# a = tf.constant(_a)
# b = tf.constant(_b)
# node = tf.tensordot(a,b,1)
#==============================================================================

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(init))
#dot_result = sess.run(node)
#print(sess.run([linear_model, W, update], feed_dict={x:[100.]}))


#==============================================================================
# if __name__ == "__main__":
#     from Block import Block
# else:
#     from .Block import Block
# import numpy as np
# from collections import deque
# from scipy.signal import hanning
# 
# class block_inc_spectrogram (Block):
#     
#     def __init__(self, _pipe, fs=250, nperseg=256, num_ch=8, window='hanning'):   
#         self.pipe = _pipe
#         self.fs = fs
#         self.nperseg = nperseg
#         self.buf = deque([np.zeros((num_ch)) for i in range(nperseg)], maxlen=nperseg)
#         if window == 'hanning':
#             self.window = hanning(nperseg)
#         else:
#             self.window = None
#         
#     def run(self, _buf, test=False):
#         '''
#         Parameters to vary:
#             window
#             segment length
#             overlap
#             detrend(I think default yes)
#             
#             buf pass in should be deque of 4 lists, with each list num_chan long
#             
#         '''
#         self.buf.extend(_buf)
#         lbuf = np.asarray(self.buf) # creates MxN matrix with M==len(self.buf), N==num_ch
#         if self.window != None:
#             lbuf = (lbuf.T * self.window).T
#         Sxx = np.abs(np.fft.fft(lbuf,axis=0))[:129]    # eliminate hard coded val for: (math.ceil(self.nperseg/2))
#         return Sxx.ravel()
#     
#     def get_axes(self):
#         return self.f
#     
#     def spectrogram_ready(self):
#         return self.captured_axes
#     
#     def get_output_dim(self, buf_len, chan_sel):
#         return 129*8 #TODO make this dynamic
#==============================================================================
    
    
    