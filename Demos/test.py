# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 18:34:54 2017

@author: marzipan
"""
import tensorflow as tf
import numpy as np


def load_test_data(fname='data001.xdf'):
    loaded_data=load_xdf(fname)
    time_series = loaded_data[0][0]['time_series']
    time_series=np.transpose(time_series)
    return time_seriess

tf.reset_default_graph()


def extract_frequency_bins(raw_input,f_start,f_end,num_bins,Fs=250):
    L=1000
    with tf.name_scope("extract_fbins"):
        tf.assert_rank(raw_input,2,message="Error extracting frequency bins, input tensor must be rank 2, #elec x #samples.")

        t_start=np.int64((L/Fs)*f_start)
        t_end=np.int64((L/Fs)*f_end)
        print(t_start)
        print(t_end)
        assert np.mod((t_end-t_start),num_bins) == 0#, "Error, cannot evenly break up frequency bins: " + (t_end-t_start) + " total bin points but " + num_bins + " requested bins."


with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    #test code start
    a=tf.constant([[0],[0]])
    b=a.get_shape()[1]

    extract_frequency_bins(a,1,2,1000)
    
    init = tf.global_variables_initializer()
    print("Global variables init", sess.run(init))

    z=sess.run(a)


#if True: writer = tf.summary.FileWriter(".\\Logs\\",sess.graph)
#sess.close()