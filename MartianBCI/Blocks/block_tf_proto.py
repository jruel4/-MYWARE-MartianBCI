# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 08:09:51 2017

@author: marzipan
"""

import tensorflow as tf

W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)

update = tf.assign(W, W*2)

linear_model = W * x + b

init = tf.global_variables_initializer()

#==============================================================================
# x = tf.placeholder(tf.complex64)
# 
# _a = [1 for i in range(256)]
# _b = [2 for i in range(256)]
# 
# a = tf.constant(_a)
# b = tf.constant(_b)
# node = tf.tensordot(a,b,1)
#==============================================================================

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(init))
#dot_result = sess.run(node)
#print(sess.run([linear_model, W, update], feed_dict={x:[100.]}))