# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from Demos.RL_Utils.Helpers import Processing_TF as Utils

class Processing_TEST(tf.test.TestCase):
    def test_2d_array(self):
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


    def test_1d_shift(self):
        x=[1,2,3,4,5,6]
        xp1=np.roll(x,1)
        xn1=np.roll(x,-1)
        xp5=np.roll(x,5)
        xn5=np.roll(x,-5)
        with self.test_session() as sess:
            x_tf=tf.constant(x)
            self.assertAllEqual(Utils.shift_1d(x_tf,0).eval(), x)
            self.assertAllEqual(Utils.shift_1d(x_tf,1).eval(), xp1)
            self.assertAllEqual(Utils.shift_1d(x_tf,-1).eval(), xn1)
            self.assertAllEqual(Utils.shift_1d(x_tf,5).eval(), xp5)
            self.assertAllEqual(Utils.shift_1d(x_tf,-5).eval(), xn5)


if __name__ == '__main__':
    tf.test.main()