# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 22:23:23 2017

@author: marzipan
"""

import tensorflow as tf
import numpy as np
from Demos.RL_Utils.Helpers import Processing_TF as Utils

class RewardExtraction:
    
    def __init__(self,_LOWALPHA=8,_HIGHALPHA=13):
        self.mHIGHALPHA=_HIGHALPHA
        self.mLOWALPHA=_LOWALPHA
        return
    
    def map_reward(self,state,electrode_weights=[],Fs=250,L=1000):
        with tf.name_scope("Generate_Reward"):

            reward = Utils.extract_frequency_bins(state,self.mLOWALPHA,self.mHIGHALPHA,1)
            reward_flat = tf.reshape(reward,[-1],name="flatten_fft")
            reward_summed = tf.reduce_sum(reward_flat,name="alphapow_summing")        
        return reward_summed
        
            #TODO Add in electrode weighting
            #    if electrode_weights == [] or len(electrode_weights) != num_elec:
            #        return tf.reduce_sum(spectro_binned[1:-2])
            #    else:
            #        e_w = tf.constant(electrode_weights)
            #        return tf.multiply(tf.reduce_sum(spectro_binned[1:-2]), electrode_weights)

    def map_reward_trivial(self, state):
        return tf.cast(tf.abs(state[0,0]),dtype=tf.float32)
        #return tf.constant(666, dtype=tf.float32)

def build_graph_reward_extraction():
#    tf.reset_default_graph()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        global z
        Sin1=[np.sin(2*np.pi * x/25) for x in range(1000)]
        
        RE=RewardExtraction()
        
        a=tf.constant([Sin1],dtype=tf.complex64)
        x = RE.map_reward(a)
    
        init=tf.global_variables_initializer()
        sess.run(init)
    
        writer = tf.summary.FileWriter(".\\Logs\\",sess.graph)    
        z=sess.run(x)
        sess.close()