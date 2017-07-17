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
from entrainment_protocol_0 import protocol_0, protocol_mod_4, create_protocol,protocol_1_follow_beta
from entrainment_protocol_1 import protocol_1, protocol_1_generator
import time

tf.reset_default_graph()

N = 60

# Tunable parameters (ALSO Learning Rate)
NUM_CELLS = 6
NUM_UNITS = 200
TIMESTEPS = 1

LEARNING_RATE=1E-7

#protocol = create_protocol(N)

LOG_DIR = 'C:\\Users\\marzipan\\workspace\\MartianBCI\\MartianBCI\\Logs\\LSTMRegressor_Model0\\' + str(time.time()) + "\\"
RNN_LAYERS = [{'num_units': NUM_UNITS} for i in range(NUM_CELLS)]
DENSE_LAYERS = [N]
TRAINING_STEPS = int(80000000)
PRINT_STEPS = TRAINING_STEPS / 10
BATCH_SIZE = 100
EEG_STATE_SIZE = 4
TRAINING_SAMPLES = 10000

#TODO add restore latest feature
LOAD_CKPT = False
path_to_restore = 'C:\\Users\\marzipan\\workspace\\MartianBCI\\MartianBCI\\Logs\\LSTMRegressor_Model0\\1499896611.7702782\\model.ckpt-9999'

# Build NN
input_x = tf.placeholder(dtype=tf.float32, shape=[None, TIMESTEPS, 0 + EEG_STATE_SIZE], name="Features")
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
        DENSE_LAYERS)

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
optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, centered=False, decay=0.8)
#TODO add learning rate print out during run
#TODO look inside with tensorboard (no more trial and error, except when done systematically...)
#optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

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

if LOAD_CKPT:
    # Restore code
    saver.restore(sess,  path_to_restore)
else:
    # Run the Op to initialize the variables.
    #init = tf.initialize_variables([train_op])
    sess.run(init)

# Weights
#==============================================================================
# bw_act = list()
# b_w = dict()
# for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
#     b_w.update({i.name: sess.run(i)})
# bw_act.append(b_w)
#==============================================================================


# Create the summary writer
summary_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
summary = tf.summary.merge_all()

# Train / Update Model

X, y = generate_data(protocol_1_generator, [i for i in range(TRAINING_SAMPLES)], TIMESTEPS, seperate=True, non_gen=False)
for k in X.keys():
    X[k] = np.squeeze(X[k], axis=1)
    y[k] = y[k].astype(np.int32)

durations = list()
size_x = list()
size_y = list()

#TODO Modify the batch feeding so that new data is passed in each time!
TRAIN_LOOP_ROLLOVER = int(len(X['train'])-BATCH_SIZE-1)
for step in range(TRAINING_STEPS):
    x_train = X['train'][(step % TRAIN_LOOP_ROLLOVER): (step % TRAIN_LOOP_ROLLOVER) + BATCH_SIZE]
    y_train = y['train'][(step % TRAIN_LOOP_ROLLOVER): (step % TRAIN_LOOP_ROLLOVER) + BATCH_SIZE]
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
    if (step + 1) % 10000 == 0:
        print("Saving checkpoint")
        checkpoint_file = LOG_DIR + 'model.ckpt'
        ckpt_path = saver.save(sess, checkpoint_file, global_step=step)
        print("Checkpoint writing: ", ckpt_path)

feed_dict_training = {
        input_x : X['val'],
        input_y : y['val']
        }

predictions, prediction_loss,lay,outy = sess.run([prediction, loss, multicellFinalState, cellOutputs],
                         feed_dict=feed_dict_training)
print("Prediction_Loss: ", prediction_loss)  
# Calculate Accuracy, Recall/Sensitivity, Specificity, Precision, F1, AUC, etc.

# predictions to one hot
hot_preds = np.zeros(y['val'].shape)
hot_preds[np.arange(hot_preds.shape[0]),predictions] = 1

import sklearn
accuracy = sklearn.metrics.accuracy_score(y['val'], hot_preds)
print("accuracy: ",accuracy)












