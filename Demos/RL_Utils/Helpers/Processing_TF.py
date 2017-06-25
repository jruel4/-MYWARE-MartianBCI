# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
        
'''
Inputs:
    raw_input: #elec x #samples tensor of raw data
    f_start: start of the first frequency bin (hz)
    f_end: end of the last frequency bin (hz)
    num_bins: total number of bins
    
    (Optional) Fs: Frequency at which the data was sampled at; defaults to 250

Returns:
    out: #elec x num_bins tensor of average power across each frequency bin
'''
def extract_frequency_bins(raw_input,f_start,f_end,num_bins,Fs=250,L=1000):
    with tf.name_scope("utils.extract_fbins"):
        tf.assert_rank(raw_input,2,message="Error extracting frequency bins, input tensor must be rank 2, #elec x #samples.")
        # Get input length
        L=raw_input.get_shape().as_list()[1]
        t_start=np.int64((L/Fs)*f_start)
        t_end=np.int64((L/Fs)*f_end)
        assert np.mod((t_end-t_start),num_bins) == 0, "Error, cannot evenly break up frequency bins: "# + (t_end-t_start) + " total bin points but " + num_bins + " requested bins."
                            
        raw_fft=tf.fft(raw_input)
        sliced_raw_fft = tf.slice(raw_fft,[0,t_start],[-1,t_end-t_start])
        sliced_raw_fft_t = tf.split(sliced_raw_fft,num_bins,axis=1)
        sliced_raw_fft_t_abs = tf.abs(sliced_raw_fft_t)
        sliced_binned_fft = tf.reduce_sum(sliced_raw_fft_t_abs,axis=2)
        sliced_binned_fft_reduced = tf.transpose(sliced_binned_fft)
    return sliced_binned_fft_reduced




'''
conv_multiple

Input:
    data - data to operate on; rank 2 #chan x #samples tensor
    kernel - rank 1 tensor; convolved over #samples for each chan
    bidir - convolve kernel both ways
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




def load_individual_baseline_test(num_elec=8,fbin_steps=10):
    individual_baselines=np.asarray(num_elec*[np.transpose([range(500,5500,500)]*fbin_steps)])
    return individual_baselines