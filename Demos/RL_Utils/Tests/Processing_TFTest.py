# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from Demos.RL_Utils.Helpers import Processing_TF as Utils

class Processing_TEST(tf.test.TestCase):
    def test_2d_array(self):
        tf.reset_default_graph()
        with self.test_session() as sess:
            expected_val = np.int64([[500,0,0,0,0],[0,0,0,0,500]])
            Sin1=[np.sin(2*np.pi * (x/250)) for x in range(1000)]
            Sin20=[np.sin(2*np.pi * (x/250) * 20) for x in range(1000)]
            
            a=tf.constant([Sin1,Sin20],dtype=tf.complex64)
            b=Utils.extract_frequency_bins(a,0,25,5)
            c=tf.cast(b,dtype=tf.int64)

            #Write graph to file
            writer = tf.summary.FileWriter(".\\Logs\\",sess.graph)
            
            #Actual check
            self.assertAllEqual(c.eval(), expected_val)
            
    def test_re_param(self):
        tf.reset_default_graph()
        with self.test_session() as sess:
            expected_val = np.int64([[0]])
#            Sin1=[np.sin(2*np.pi * (x/250)) for x in range(1000)]
            Sin20=[np.sin(2*np.pi * (x/250) * 20) for x in range(1000)]
            
            a=tf.constant([Sin20],dtype=tf.complex64)
            b=Utils.extract_frequency_bins(a,8,13,1)
            c=tf.cast(b,dtype=tf.int64)

            #Write graph to file
            writer = tf.summary.FileWriter(".\\Logs\\",sess.graph)
            
            #Actual check
            self.assertAllEqual(c.eval(), expected_val)

if __name__ == '__main__':
    tf.test.main()