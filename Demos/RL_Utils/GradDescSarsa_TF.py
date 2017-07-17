'''
Top level class for Gradient Descent Sarsa (Lambda) RL Algorithm

Usage: Input a 'state', output as 'action'. 

E.g. input eeg data, output audio commands.

TODO Entire tensorflow session shall be encapsulated by this class
'''

import tensorflow as tf

class GradDescSarsaAgent:
    
    def __init__(self):
        pass

    def build_main_graph(self):

        with tf.variable_scope("root",reuse=True):
            
            # Graph Inputs
            raw_data_new = tf.placeholder(tf.complex64,name='p_raw_data_new')
                
            # Constants
            epsilon = tf.constant(_epsilon, dtype=tf.float32,name='c_epsilon')
            alpha = tf.constant(_alpha, dtype=tf.float32,name='c_alpha')
            gamma = tf.constant(_gamma, dtype=tf.float32,name='c_gamma')
            lam = tf.constant(_lambda, dtype=tf.float32,name='c_lambda')
        
            # Variables
            weights = tf.get_variable(name="v_weights",shape=[feat_space_size],dtype=tf.float32,initializer=tf.constant_initializer(0))
            z_trace = tf.get_variable(name="v_ztrace",shape=[feat_space_size],dtype=tf.float32,initializer=tf.constant_initializer(0))
            expected_reward = tf.get_variable(name="v_expected_reward",shape=[],dtype=tf.float32,initializer=tf.constant_initializer(0))
            
            #STEP 1: Calculate actual reward
            actual_reward = map_reward(raw_data_new)
        
            #STEP 2: Find the error between expected and actual
            err_delta = tf.subtract(actual_reward, expected_reward, name="o_calc_err_delta")
        
            #STEP 3: Determine if we're greedy or not    
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
            z_trace = tf.minimum(z_tmp,tf.ones(z_tmp.get_shape()),name="z_trace_replace")
            
        return raw_data_new, action_next
    
    def interact(self, state):
        '''
        Purpose: top level 
        '''
        action = None
        return action
    
    
    