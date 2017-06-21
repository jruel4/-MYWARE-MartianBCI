# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 22:23:23 2017

@author: marzipan
"""

import tensorflow as tf
import numpy as np
from Demos.RL_Utils.Helpers import RewardExtraction_TF as RewardExtraction

class RewardExtraction_TEST(tf.test.TestCase):
    def test_1d_array(self):
        with self.test_session() as sess:
            expected_val_1 = np.int64(499) #TODO this should be 500?
            expected_val_2 = np.int64(0)

            Sin10=[np.sin(2*np.pi * (x/250) * 10) for x in range(1000)]
            Sin20=[np.sin(2*np.pi * (x/250) * 20) for x in range(1000)]

            RE=RewardExtraction.RewardExtraction()
            a=tf.constant([Sin10],dtype=tf.complex64)
            b=tf.constant([Sin20],dtype=tf.complex64)
            in_spect = RE.map_reward(a)       
            in_spect = tf.cast(in_spect,dtype=tf.int64)
            out_spect = RE.map_reward(b)
            out_spect = tf.cast(out_spect,dtype=tf.int64)
            
            #Write graph to file
            writer = tf.summary.FileWriter(".\\Logs\\",sess.graph)
            
            #Actual check
            self.assertAllEqual(in_spect.eval(), expected_val_1)
            self.assertAllEqual(out_spect.eval(), expected_val_2)

    def test_2d_array(self):
        with self.test_session() as sess:
            expected_val_1 = np.int64(500)

            Sin10=[np.sin(2*np.pi * (x/250) * 10) for x in range(1000)]
            Sin20=[np.sin(2*np.pi * (x/250) * 20) for x in range(1000)]

            RE=RewardExtraction.RewardExtraction()
            a=tf.constant([Sin10,Sin20],dtype=tf.complex64)
            in_spect = RE.map_reward(a)       
            in_spect = tf.cast(in_spect,dtype=tf.int64)
            
            #Write graph to file
            writer = tf.summary.FileWriter(".\\Logs\\",sess.graph)
            
            #Actual check
            self.assertAllEqual(in_spect.eval(), expected_val_1)
 
if __name__ == '__main__':
    tf.test.main()
