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


def load_individual_baseline_test(num_elec=8,fbin_steps=10):
    individual_baselines=np.asarray(num_elec*[np.transpose([range(500,5500,500)]*fbin_steps)])
    return individual_baselines