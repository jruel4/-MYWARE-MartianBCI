# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 18:34:54 2017

@author: marzipan
"""
import tensorflow as tf
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from Demos.RL_Utils.Helpers.LoadXDF import get_raw_data

def load_test_data(fname='data001.xdf'):
    loaded_data=load_xdf(fname)
    time_series = loaded_data[0][0]['time_series']
    time_series=np.transpose(time_series)
    return time_series





'''
conv_multiple

Input:
    data - data to operate on; rank 2 #chan x #samples tensor
    kernel - rank 1 tensor; convolved over #samples for each chan
    kernel_len - length of the kernel
    !NOTE! kernel must be <= 2nd dimension of data (#samples)
    
Out:
    convolved - rank 2 #chan x #samples tensor
    
Function:
    Convolves kernel with each input channel - useful for applying a single filter to multiple channels

'''

def multi_ch_conv(data,kernel,bidir=True):
    #Verify input tensor ranks
    tf.assert_rank(kernel,1)
    tf.assert_rank(data,2)
 
    num_channels = tf.shape(data,name="NumChan")[0]
    num_samples = tf.shape(data,name="NumSamples")[1]
    kernel_len = tf.shape(kernel,name="KernelLen")[0]
    
    '''
    NOTE
    The above assert_rank's are evaluated when the grab is being compiled.
    The below assert_less_equal is evaluated when the graph is run, so it
    must actually be *run*, which is why we have to use the control_dependencies
    call below.
    '''

    asserts= [
            tf.assert_less_equal(kernel_len, num_samples, message="JCR: Lenth of kernel must be shorter than the length of the input.")
            ]

    with tf.control_dependencies(asserts):
 
        #Pad the beginning / end of each channel so we can do a single 1D convolve
        with tf.name_scope("PadInputTensor"):
            p = tf.cast( tf.ceil( tf.realdiv(tf.cast(kernel_len,tf.float32) , tf.constant(2.0,dtype=tf.float32) ) ), dtype=tf.int32 )
            data_pad = tf.pad(data,[[0,0],[p,p]])
        
        with tf.name_scope("Conv1D"):
            #Reshape to use with conv1d
            #[batch,width,chan]
            data_1d = tf.reshape(data_pad,[1,-1,1])
            
            #Reshape kernel to use with conv1d
            #[width,chan_in,chan_out]
            kernel_1d = tf.reshape(kernel,[-1,1,1])
        
            conv_raw = tf.nn.conv1d(data_1d,kernel_1d,1,'SAME')
            
            if bidir:
                conv_raw_rev = tf.nn.conv1d( tf.reverse(conv_raw,[1]), kernel_1d,1,'SAME' )
                conv_raw = tf.reverse(conv_raw_rev,[1])

        with tf.name_scope("Reconstruct"):
            conv_raw_rs = tf.reshape(conv_raw,[num_channels,-1])
        
            conv_raw_sliced = tf.slice(conv_raw_rs,[0,p],[-1,num_samples])
    
        return conv_raw_sliced

tf.reset_default_graph()


def SinHz(hz,Fs=250,L=1000):
    return [np.sin(2*np.pi* x * hz / Fs) for x in range(L)]

L=2000
hz=4
Fs=250

fir_coeffs = '8-13_BP_FIR_BLACKMANN.npy'
b,a = np.load('C:\\Conda\\MartianBCI\\Demos\\RL_Utils\\Filters\\' + fir_coeffs)
n_order = len(b)


pad=np.ceil(n_order/2.0)


#rd=get_raw_data()[0]

if True:
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    
        
        #test code start
        sin=((np.add(SinHz(50,Fs,L), SinHz(10,Fs,L))))
#        p=np.concatenate((sin,sin))
#        t_a=tf.constant(a)
#        print(t_a)
#        y=tf.cast(tf.reshape(t_a,[1,2000,1]),tf.float32)
#        print(t_a)
#        t_coef1=tf.constant(q)
#        x=tf.cast(tf.reshape(t_coef1,[511,1,1]),tf.float32)
        
#        t_b = tf.nn.conv1d(y,x,1,'SAME')
#        z=sess.run(t_b)
        
        if False:        
            plt.plot(np.linspace(0,L/Fs,L), np.add(SinHz(50,Fs,L), SinHz(10,Fs,L)))
            plt.figure()
#            c=signal.filtfilt(b,a,np.add(SinHz(50,Fs,L), SinHz(10,Fs,L)))
#            plt.plot(np.linspace(0,L/Fs,L), c)
#            plt.figure()
#            plt.plot(np.linspace(0,L/Fs,L), z[0])
#            plt.figure()

        
#        b=multi_ch_conv(
#                tf.constant([[1,1,1,1],[2,2,2,2]],dtype=tf.float32),
#                tf.constant([3,3,3,3,3,3],dtype=tf.float32),
#                True
#                )
        b=multi_ch_conv(
                tf.constant( np.asarray(np.transpose(rd)), dtype=tf.float32),
                tf.constant( b, dtype=tf.float32)
                )
        init = tf.global_variables_initializer()
        print("Global variables init", sess.run(init))
        writer = tf.summary.FileWriter(".\\Logs\\",sess.graph)
        e=sess.run(b)
        
#        plt.plot(e[1])
#        plt.figure()
#        plt.show()    
    #if True: writer = tf.summary.FileWriter(".\\Logs\\",sess.graph)
    #sess.close()

