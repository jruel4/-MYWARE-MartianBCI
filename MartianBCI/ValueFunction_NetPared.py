
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf
import timeit
import time
from scipy import signal
from pylsl import  StreamInlet, resolve_stream
from queue import Queue, Empty, Full
from threading import Thread, Event


import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils as TFUtils

# Regressor Model 0 Hyperparameters
G_dense0_size = 10
G_dense1_size = 10


'''
MODEL
'''
# Unpack the input features and labels
state = tf.placeholder(tf.float32, shape=[None, 1])
print(state)

# Dense Layer 1
dense0 = tf.layers.dense(inputs=state,
                        units=G_dense0_size,
                        kernel_initializer=tf.constant_initializer(1),
                        activation=tf.sigmoid)
print(dense0)

# Dense Layer 1
dense1 = tf.layers.dense(inputs=dense0,
                        units=G_dense1_size,
                        kernel_initializer=tf.constant_initializer(1),
                        activation=tf.sigmoid)
print(dense0)

# Add a final, fully connected layer which will represent our output state
output_layer = tf.layers.dense(inputs=dense1,
                               units=1,
                               activation=None)
print(output_layer)



# Loss
lbl=tf.placeholder(tf.float32,[None,1])
loss = tf.losses.mean_squared_error(
    labels=lbl,
    predictions=output_layer
    )

# Train Ops

# Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.GradientDescentOptimizer(5e-2)

# Create a variable to track the global step.
global_step = tf.Variable(0, name='global_step', trainable=False)

# Use the optimizer to apply the gradients that minimize the loss
# (and also increment the global step counter) as a single training step.
train_op = optimizer.minimize(loss, global_step=global_step)

# Add the variable initializer Op.
init = tf.global_variables_initializer()

# Create a session for running Ops on the Graph.
sess = tf.Session()

# Run the Op to initialize the variables.
sess.run(init)


n_len=10
s=0
func_tst = np.asarray([[x,x**2] for x in np.linspace(0,5,n_len)])
for step in range(10000):
    
    s = (s+1) % n_len
    nn_input=np.asarray([[ func_tst[s,0] ]])
    reward = np.asarray([[ func_tst[s,1] ]])
    
    # Setup the feed dict for the loss function
    feed_dict_training = {
            state: nn_input,
            lbl:reward
        }
    
    # Train / Update Model
    start_time = time.time()
    _, loss_value = sess.run([train_op, loss],
                             feed_dict=feed_dict_training)

    train_duration = time.time() - start_time
    print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, train_duration))

b_w = dict()
bw_act = list()
for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    b_w.update({i.name: sess.run(i)})
bw_act.append(b_w)

        
        
        
        
        