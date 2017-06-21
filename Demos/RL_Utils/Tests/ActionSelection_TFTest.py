# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 22:23:54 2017

@author: marzipan
"""

import tensorflow as tf
import numpy as np

from Demos.RL_Utils.Helpers import ActionSelection_TF as ActionSelection
from Demos.RL_Utils.Helpers import BinaryFeatureExtraction_TF as BinaryFeatureExtractor
from Demos.RL_Utils.Helpers import RewardExtraction_TF as RewardExtraction


class ActionSelection_TEST(tf.test.TestCase):
    
    
    def greedy_action_TEST(self):
        with self.test_session():
            best_action = 5
            
            AMP = np.linspace(0, 1, 5)
            BBF = np.linspace(0, 20, 21)
            CF = np.linspace(250,1000, 5)
               
            feat_extract = BinaryFeatureExtractor.BinaryFeatureExtractor([21,3])
            reward_extract = RewardExtraction.RewardExtraction()

            act_sel = ActionSelection.ActionSelection(feat_extract,reward_extract,AMP,BBF,CF)
            
            weights = feat_extract.activateBinaryFeatures_TF(best_action)
            act = act_sel.greedy_action(0,weights)
            
            self.assertAllEqual(best_action,act)
        return
            
    def random_action_TEST(self)
        with self.test_session():
            best_action = 5
            
            AMP = np.linspace(0, 1, 5)
            BBF = np.linspace(0, 20, 21)
            CF = np.linspace(250,1000, 5)
               
            feat_extract = BinaryFeatureExtractor.BinaryFeatureExtractor([21,3])
            reward_extract = RewardExtraction.RewardExtraction()

            act_sel = ActionSelection.ActionSelection(feat_extract,reward_extract,AMP,BBF,CF)
            
            for i in 10000:
                
                
                
if __name__ == '__main__':
  tf.test.main()

   
    
    
    
'''    
    def __init__(self):
        self.mAS = ActionSelection()
    
    def map_next_action_TEST(self):
        self.mAS.map_next_action([[0]*1000],[1,0,0],[[0]*],epsilon)

    def greedy_action_TEST(self,raw_input,weights):
    
    
    def exploratory_action_TEST(self):
    
    # in: x, length 3 tensor of amp x bbf x cf
    def act_to_actbin_idx_TEST(self,x):
    def act_to_actbin_TEST(self):
    def actbin_idx_to_act_TEST(self,idx):
    def actbin_to_act_TEST(self,actbin):
    
    
    
    
  def testSquare(self):
    with self.test_session():
      bfe = BinaryFeatureExtractor()
      bfe.initBinaryFeatureList(21, 3)
      y = bfe.activateBinaryFeatures_TF(10)
      y_pythonic = bfe.activateBinaryFeaturesBrute(10)
      self.assertAllEqual(y.eval(), y_pythonic)

if __name__ == '__main__':
  tf.test.main()
'''