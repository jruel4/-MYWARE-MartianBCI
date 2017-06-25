# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from Demos.RL_Utils.Helpers.LoadXDF import get_raw_data
from Demos.RL_Utils.Helpers.Processing_TF import multi_ch_conv
from Demos.RL_Utils.Helpers.Processing_TF import fft_cpu

#[ts,locs] = get_raw_data()

#raw = tf.constant(ts)

fir_coeffs = '7-14_BP_FIR_BLACKMANN.npy'
b,a = np.load('C:\\Conda\\MartianBCI\\Demos\\RL_Utils\\Filters\\' + fir_coeffs)

#rdd=get_raw_data()[0]
rd=rdd[2500:5600]

def SinHz(hz,Fs=250,L=1000):
    return [np.sin(2*np.pi* x * hz / Fs) for x in range(L)]

if True:
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        
        raw_data_tensor = tf.constant( np.asarray(np.transpose(rd)), dtype=tf.float32)
        b_coeffs = tf.constant( b, dtype=tf.float32)

        data_bp_filt_tmp=multi_ch_conv(raw_data_tensor,b_coeffs)
        data_bp_filt=tf.slice(data_bp_filt_tmp, [0,300], [-1,tf.shape(data_bp_filt_tmp)[1]-600])

        pphz = tf.realdiv( tf.cast(tf.shape(data_bp_filt)[1], tf.float32) , tf.constant(250.0))

        #This value (in Hz) is used to determine the peak frequency - larger window uses more surrounding values to calculate IAF
        IAF_Peak_Window_Size = tf.constant(1.0 / 2.0)
        IAF_Window = tf.ones([tf.cast(tf.multiply(IAF_Peak_Window_Size,pphz),tf.int32)])

        data_fft = fft_cpu([data_bp_filt[1]],1,2500)
        
        windowd = multi_ch_conv(tf.cast(tf.abs(data_fft),tf.float32),IAF_Window)
        
        init = tf.global_variables_initializer()
        print("Global variables init, ret: ", sess.run(init))
        writer = tf.summary.FileWriter(".\\Logs\\",sess.graph)
        e=sess.run([data_bp_filt,data_bp_filt_tmp,data_fft,windowd])
        
        
        


#L=1000



#b=np.transpose()
#b=SinHz(5)

#c=np.matmul(a,b)