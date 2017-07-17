# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 18:34:54 2017

@author: marzipan
"""
import tensorflow as tf
import numpy as np

act_space_size=6400

def act_to_actbin_idx(x):
#    print("x2: ",x)
    return (x[0]) + (x[1]*10) + (x[2]*10*32)

def act_to_actbin(x):
#    print("x1: ",x)
    idx=act_to_actbin_idx(x)
    #FIX
    y = tf.SparseTensor([[tf.cast(idx,dtype=tf.int64)]],[True],[act_space_size])
#    print('Y', y)
    return tf.sparse_tensor_to_dense(y,default_value=False)


action_old = tf.Variable(tf.ones([3]),name="v_next_indices")
action_next = tf.Variable(tf.ones([3]),name="v_action_next_indices")

action_old=action_next
a=act_to_actbin(action_old)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

init = tf.global_variables_initializer()
print("Global variables init", sess.run(init))


b=sess.run(a)
if True: writer = tf.summary.FileWriter(".\\Logs\\",sess.graph)
sess.close()