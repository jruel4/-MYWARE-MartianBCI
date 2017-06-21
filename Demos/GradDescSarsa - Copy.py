# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 12:43:38 2017

@author: marzipan
"""

#ROSHI uses phase of signals
#Different signals for each hemisphere

#Minimum delay is 150ms-250ms

###SMR
#The argument goes as follows: A narrow-band filter
#can be seen for our purposes as a transducer of frequency fluctuations into
#amplitude fluctuations. Frequency variation and phase fluctuations are obviously
#directly related. Dynamic, continuous reward-based training using narrow-band
#filters attempts to shape the EEG frequency distribution toward the middle of the
#resonance curve, with often immediate and sometimes trenchant consequences for
#the person’s state. These factors are in play even in ostensibly single-site amplitude
#training with referential placement (because references are not silent). 

#Bipolar montage then further augments the role of narrow-band filters as phase
#discriminants because the amplitudes at the two sites are now more correlated than in
#referential montage, which shifts the burden of variability more onto the phase. In
#typical application, the bipolar montage will be deployed either at near-neighbor
#sites or at homotopic sites. In these cases, the correlation of amplitudes (i.e.,
#comodulation) is typically enhanced with respect to arbitrary site combinations. 

                                                                             
###ALPHA
#In Les Fehmi’s mechanization of synchrony training, the reinforcement is
#delivered with every cycle of the alpha rhythm that meets criterion. It turns out
#that the timing of the delivery of the reward signal with respect to the underlying
#alpha signal is crucial. With the phase delay optimized, the reward pulse serves to
#augment the next cycle of the alpha spindle. This is firstly another demonstration
#that “phase matters. ” Secondly, we have here a stimulation aspect to what is fundamentally
#a feedback paradigm. 
                                                                             
#Record and *TAG* all data - we can look for correlations offline



import tensorflow as tf
import time
import numpy as np
from matplotlib import pyplot as plt

from RL_Environment_Simulator import RLEnv

#Clear graph, for tensorboard (and hygiene)
#tf.reset_default_graph()


##GLOBAL SETTINGS
Fs=250



fbin_min = 0.5
fbin_max = 50.0
fbin_steps = 2
#fbin_stepsize = 0.25
fbins = np.concatenate(([0.0], np.linspace(fbin_min,fbin_max,fbin_steps), [Fs/2.0]))


electrode_cnt = 8
feat_per_fbin_per_ch = 10
recording_len = 1000 #in samples

state_space_size = feat_per_fbin_per_ch*fbin_steps*electrode_cnt
act_space_size = amp_steps*bbf_steps*cf_steps
feat_space_size = state_space_size + act_space_size


def tf_build_graph(tf_session,global_init,_user_fbin_baselines):
    with tf.variable_scope("root",reuse=True):
        
        # Graph Inputs
        raw_data_new = tf.placeholder(tf.complex64,name='p_raw_data_new')
            
        # Constants
        epsilon = tf.constant(_epsilon, dtype=tf.float32,name='c_epsilon')
        alpha = tf.constant(_alpha, dtype=tf.float32,name='c_alpha')
        gamma = tf.constant(_gamma, dtype=tf.float32,name='c_gamma')
        lam = tf.constant(_lambda, dtype=tf.float32,name='c_lambda')
        user_fbin_baselines=tf.constant(_user_fbin_baselines,dtype=tf.float32,name='c_fbin_baselines')
    
        weights = tf.get_variable(name="v_weights",shape=[feat_space_size],dtype=tf.float32,initializer=tf.constant_initializer(0))
        z_trace = tf.get_variable(name="v_ztrace",shape=[feat_space_size],dtype=tf.float32,initializer=tf.constant_initializer(0))
        expected_reward = tf.get_variable(name="v_expected_reward",shape=[],dtype=tf.float32,initializer=tf.constant_initializer(0))
        
        #STEP 1: Calculate actual reward
        #TODO write unit tests for all sub routines
        actual_reward = map_reward(raw_data_new)
    
        #STEP 2: Find the error between expected and actual
        err_delta = tf.subtract(actual_reward, expected_reward, name="o_calc_err_delta")
    
        #STEP 3: Determine if we're greedy or not    
        #TODO eliminate exploratory behavior for initial testing
        action_next = map_next_action(raw_data_new,0,weights,epsilon)
    
        #STEP 5: Generate new S/A features
        bin_features = map_features_binary(raw_data_new,action_next,user_fbin_baselines)
        #Cast to float32 for easy operation
        bin_features = tf.cast(bin_features,dtype=tf.float32)
        
        #STEP 6: Generate new expected reward (used to update w,z,del)
        er = tf.reduce_sum(tf.multiply(weights,bin_features))
        expected_reward = er
    
        #STEP 7: Update our model
        err_delta = err_delta + gamma * expected_reward
        weights = weights + alpha * err_delta * z_trace
        z_tmp = (z_trace * gamma * lam) + bin_features #Decay, and update
        z_trace = tf.minimum(z_tmp,tf.ones(z_tmp.get_shape()),name="z_trace_replace") #uncomment to make replacing trace
    
        
        
    return raw_data_new,[action_next,summaries]

state=None
next_action=None
out=None
def main(tf_sess,global_init=True):    
    global state
    global out
    #
    
    
    agent=RLEnv()
 




    
    next_action=(bbf[0],amp[0])
    for i in range(10):
        try:

        except KeyboardInterrupt:
            return
    return


with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    main(sess,global_init=True)
    sess.close()

if False:
    b=np.asarray(b)
    plt.plot(b[:,3])