# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:20:21 2017

@author: marzipan
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.contrib import learn
from sklearn.metrics import mean_squared_error
from basic_lstm import lstm_model
from data_processing import generate_data
from entrainment_protocol_0 import protocol_0, protocol_mod_4, create_protocol
import time

tf.reset_default_graph()

N = 60

# Tunable parameters (ALSO Learning Rate)
NUM_CELLS = 2
NUM_UNITS = 50
TIMESTEPS = 15

LEARNING_RATE=1E-3

#protocol = create_protocol(N)

LOG_DIR = 'C:\\Users\\marzipan\\workspace\\MartianBCI\\MartianBCI\\Logs\\LSTMRegressor_Model0\\' + str(time.time()) + "\\"
RNN_LAYERS = [{'num_units': NUM_UNITS} for i in range(NUM_CELLS)]
DENSE_LAYERS = [N]
TRAINING_STEPS = 50
PRINT_STEPS = TRAINING_STEPS / 10
BATCH_SIZE = 5000

    
#X, y = generate_data(np.sin, np.linspace(0, 100, 10000, dtype=np.float32), TIMESTEPS, seperate=False)
'''
format of data is X = num_samples x num_timesteps x input_dim
                  y = num_samples x output_dim

given that both X and y are derived from a time series:
    X1 = [v0,v1,v2], y1 = [v3]
    X2 = [v1,v2,v3], y2 = [v4]
    

'''
X, y = generate_data(protocol_0, [i for i in range(10000)], TIMESTEPS, seperate=False, non_gen=True)

for k in X.keys():
    #X[k] = X[k].astype(np.int32)
    y[k] = y[k].astype(np.int32)


# Build NN
input_x = tf.placeholder(dtype=tf.float32, shape=[None, TIMESTEPS, N], name="Features")
input_y = tf.placeholder(dtype=tf.float32, shape=[None, N], name="Labels")

# RNN Cell Ops
LSTMCellOps = [tf.contrib.rnn.BasicLSTMCell(layer['num_units'], state_is_tuple=True,forget_bias=2.0) for layer in RNN_LAYERS]

stackedLSTM = tf.contrib.rnn.MultiRNNCell(LSTMCellOps, state_is_tuple=True)
unstackedInput = tf.unstack(input_x, axis=1, num=TIMESTEPS, name="UnrolledFeatures")


# cellOutputs corresponds to the output of each multicell in the unrolled LSTM
# finalState maps to last cell in unrolled LSTM; has one entry for each cell in the multicell ([C0,...,Cn] for an n-celled multicell), each entry is tuple corresponding to (internalHiddenState,outputHiddenState)
cellOutputs, multicellFinalState = tf.contrib.rnn.static_rnn(stackedLSTM, unstackedInput, dtype=tf.float32)

# Connect a dense layer to the output of the last cell in the unrolled LSTM
dense_out = tf.contrib.layers.stack(
        cellOutputs[-1],
        tf.contrib.layers.fully_connected,
        DENSE_LAYERS,
        name="Dense")

# Prediction and loss metrics
prediction = tf.argmax(dense_out,1)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=dense_out,
        labels=input_y,
        name = 'SoftmaxOutput'))


# Define Summaries
tf.summary.scalar('loss', loss)
tf.summary.scalar('prediction', prediction[0])

# TRAINING OPS

# Create the gradient descent optimizer with the given learning rate.
optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE)

# Create a variable to track the global step.
global_step = tf.Variable(0, name='global_step', trainable=False)

# Use the optimizer to apply the gradients that minimize the loss
# (and also increment the global step counter) as a single training step.
train_op = optimizer.minimize(loss, global_step=global_step)


# Add the variable initializer Op.
init = tf.global_variables_initializer()

# Create a saver for writing training checkpoints.
saver = tf.train.Saver()

# Create a session for running Ops on the Graph.
sess = tf.Session()

# Create the summary writer
summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
summary = tf.summary.merge_all()

# Run the Op to initialize the variables.
sess.run(init)

        
# Train / Update Model
durations = list()
size_x = list()
size_y = list()
for step in range(TRAINING_STEPS):
    x_train = X['train'][(step % 5000): (step % 5000) + BATCH_SIZE]
    y_train = y['train'][(step % 5000): (step % 5000) + BATCH_SIZE]
    feed_dict_training = {
            input_x : x_train,
            input_y : y_train
            }
    size_x.append(x_train.shape)
    size_y.append(x_train.shape)
    start_time = time.time()
    _, loss_value = sess.run([train_op, loss],
                             feed_dict=feed_dict_training)
    durations.append(time.time() - start_time)

    # Write the summaries and print an overview fairly often.
    if step % 100 == 0:
        print('Step %d: loss = %.10f (%.3f sec)' % (step, loss_value, durations[-1]))
        summaries = sess.run(summary, feed_dict=feed_dict_training)
        summary_writer.add_summary(summaries, step)
        summary_writer.flush()
    
    # Save a checkpoint periodically.
    if (step + 1) % 500 == 0:
        print("Saving checkpoint")
        checkpoint_file = LOG_DIR + 'model.ckpt'
        saver.save(sess, checkpoint_file, global_step=step)
    #        print("Checkpoint writing: ", time.time() - a)

def predict(shift):
    x_pred = X['val'][shift:shift+1]
    y_pred = y['val'][shift:shift+1]
    feed_dict_training = {
            input_x : x_pred,
            input_y : y_pred
            }
    global lay, outy
    predictions, prediction_loss,lay,outy = sess.run([prediction, loss, multicellFinalState, cellOutputs],
                             feed_dict=feed_dict_training)
    print("Prediction_Loss: ", prediction_loss)    

for i in range(10): predict(i)





