# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 22:23:54 2017

@author: marzipan
"""

import tensorflow as tf
import numpy as np

from Demos.RL_Utils.Helpers import BinaryFeatureExtraction_TF as BinaryFeatureExtractor
from Demos.RL_Utils.Helpers import RewardExtraction_TF as RewardExtraction

class ActionSelection:

    def __init__(self, _FeatureExtractor, _RewardGenerator, _amp, _bbf, _cf):
        
        #Checking to ensure these are vectors
        assert np.linalg.matrix_rank(_amp) == 1
        assert np.linalg.matrix_rank(_bbf) == 1
        assert np.linalg.matrix_rank(_cf) == 1
        
        self.mFeatureExtractor = _FeatureExtractor
        self.mRewardGenerator = _RewardGenerator
        
        self.mAmpSteps = len(_amp)
        self.mBbfSteps = len(_bbf)
        self.mCfSteps = len(_cf)
        
        self.mAmp = _amp
        self.mBbf = _bbf
        self.mCf = _cf
        
        return
    
    def map_next_action(self,raw_input,action,weights,epsilon):
        with tf.name_scope("MapNextAction"):
            random = tf.random_uniform([1])
            next_action = tf.cond(random[0] > epsilon,
                                  lambda: self.greedy_action(raw_input,weights),
                                  lambda: self.exploratory_action())
        return next_action
    

    def greedy_action(self,raw_input,weights):
        feature_matrix = self.mFeatureExtractor.getBinaryFeatureFullMatrix_TF()
        action_values = tf.matmul(weights,tf.cast(feature_matrix,tf.float32),False,True)
        next_action = tf.arg_max(action_values,1)
        return next_action
    
    
    def exploratory_action(self):
#        arand= tf.random_uniform([1],minval=0,maxval=np.int64(self.mAmpSteps), dtype=tf.int64)
        brand= tf.random_uniform([1],minval=0,maxval=np.int64(self.mBbfSteps), dtype=tf.int64)
#        crand= tf.random_uniform([1],minval=0,maxval=np.int64(self.mCfSteps), dtype=tf.int64)
        return brand
    
global z
def build_graph_action_selection():
    ## USER VARIABLES: FEEL FREE TO ADJUST ##
    amp_min=0
    amp_max = 1.0 #between 0 and 1.0, corresponds to computer max volume
    amp_steps = 5
    #amp_stepsize=0.1
    amp = np.linspace(amp_min,amp_max,amp_steps)
    
    bbf_min=1.0
    bbf_max=19.0
    bbf_steps=20
    #bbf_stepsize=0.5
    bbf = np.linspace(bbf_min, bbf_max, bbf_steps)
    
    cf_min=250.0
    cf_max=1000.0
    cf_steps=5
    #cf_stepsize=50.0
    cf = np.linspace(cf_min, cf_max, cf_steps)
    
    
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    
        bfe = BinaryFeatureExtractor.BinaryFeatureExtractor()
        bfe.initBinaryFeatureList(21, 3)
        re = RewardExtraction.RewardExtraction()
        act_sel = ActionSelection(bfe,re,amp,bbf,cf)

        #2D Tensor b/c used with matmul
        weights=tf.constant([[0 if i != 10 else 5 for i in range(42)]])
        
        a=act_sel.greedy_action(0,weights)
        b=act_sel.exploratory_action()

        init=tf.global_variables_initializer()
        sess.run(init)
        writer = tf.summary.FileWriter(".\\Logs\\",sess.graph)    
        z=sess.run([a,b])
        print(z)
        sess.close()
        return z
        
#a=build_graph_action_selection()


