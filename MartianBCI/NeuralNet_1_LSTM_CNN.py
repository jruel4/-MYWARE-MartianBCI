#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib



'''
inputs:
    ?
outputs:
    returns [DATA, LABEL] where DATA is previous data fft and LABEL is current fft
'''
def input_fn():
    return

'''
inputs:
    ?
outputs:
    returns [BATCH x [DATA, LABEL]] where DATA is previous data fft and LABEL is current fft, BATCH is the batch number
'''
def input_fn_batch():
    return



#GLOBALS
G_FS = 250
G_SIGLEN = 1000
G_NCHAN = 8
G_BINSIZE = 0.5
G_NBINS = G_SIGLEN // np.maximum(G_BINSIZE // (G_FS / G_SIGLEN), 1.0)



tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.INFO)




G_FS=250
G_NCHAN=16
G_NSAMPLES=500

#PARAMETERS
G_WINDOW_LEN = 200
G_WINDOW_STRIDE = 5
G_SPECGRAM_LEN = (G_NSAMPLES - G_WINDOW_LEN) // G_WINDOW_STRIDE + 1

cnn0_num_kernels = 32
cnn0_kernel_height =    5
cnn0_kernel_width =     20
cnn0_kernel_stride =    1

print("Kernel temporal length: ",
      ((cnn0_kernel_height-1) * G_WINDOW_STRIDE + G_WINDOW_LEN) / G_FS)
print("Kernel spectral length: ",
      ((G_FS / G_WINDOW_LEN) * cnn0_kernel_width))
print("Total number of convolutions: ",
      (G_SPECGRAM_LEN / cnn0_kernel_stride),
      " x ",
      (G_WINDOW_LEN / cnn0_kernel_stride),
      " = ",
      (G_SPECGRAM_LEN / cnn0_kernel_stride) * (G_WINDOW_LEN / cnn0_kernel_stride) * cnn0_num_kernels)

print()
print("Neural net input shape: \n",
      G_SPECGRAM_LEN, " x ",  G_WINDOW_LEN, " x ", G_NCHAN, "\n",
      "Total size of: ", G_SPECGRAM_LEN * G_WINDOW_LEN * G_NCHAN)


#Setup a few variables for ease of readability
cnn0_kernel = [cnn0_kernel_height, cnn0_kernel_width]




def cnn_model_EEG(specgram, actual_next_state, mode):
    """Model function for CNN."""
    
    '''
    We assume that our input is a spectrogram of shape:
        nwindows x nchan x window_len
    
    Typical dimensions might be:
        100 x 16 x 250
    Indicating 100, 250 length spectrograms for 16 seperate channels
    '''

    # Input Reshaping
    # Input Tensor Shape: [batch_size, n_win, n_ch, w_len]
    # Output Tensor Shape: n_ch * [batch_size, n_win, w_len, 1]
    specgram_4d=tf.expand_dims(tf.abs(specgram),0)
    specgram_NWHC=tf.transpose(specgram_4d,[0,1,3,2])
    # We split this into seperate tensors for each channel b/c channels are not directly
    # related, at least in current model
    input_layer = tf.split(specgram_NWHC, G_NCHAN, axis=3)
    
    '''
    Note that CNN input has shape:
        1 x num_windows x window_len x channels
        
    Importantly, spectrograms are:
        num_windows x window_len
    
    Typical values would be something like:
        10 x 200
    
    Keep this in mind when setting kernel parameters.
            
    '''
    
    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: n_ch * [batch_size, n_win, w_len, 1]
    # Output Tensor Shape: n_ch * [batch_size, n_win, w_len, 32]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        stride=[1,1],
        padding="SAME",
        activation=tf.nn.relu)
    
    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: n_ch * [batch_size, n_win, w_len, 32]
    # Output Tensor Shape: n_ch * [batch_size, n_win/2, w_len/2, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    
    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: n_ch * [batch_size, n_win/2, w_len/2, 32]
    # Output Tensor Shape: n_ch * [batch_size, n_win/2, w_len/2, 64]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    
    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: n_ch * [batch_size, n_win/2, w_len/2, 64]
    # Output Tensor Shape: n_ch * [batch_size, n_win/4, w_len/4, 64]
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: n_ch * [batch_size, n_win/2, w_len/2, 64]
    # Output Tensor Shape: [batch_size, n_win/4 * w_len/4 * 64 * n_ch]
    pool2_cat = tf.concat(pool2,axis=0)
    pool2_flat = tf.reshape(pool2_cat, [-1, (G_SPECGRAM_LEN/4) * (G_WINDOW_LEN/4) * 64 * G_NCHAN])
    
    
    # JCR
    # Dense Layer
    # Densely connected layer with 65536 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=500, activation=tf.nn.relu)
    
    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)
    
    
    # JCR
    # Add a final, fully connected layer which will represent our output state
    predicted_next_state = tf.layers.dense(inputs=dropout,
                                           units=G_NCHAN * G_WINDOW_LEN,
                                           activation=tf.nn.relu)
    
    loss = None
    train_op = None
    
    # Calculate Loss (for both TRAIN and EVAL modes)
    '''
    JCR NOTE:
        MSE would appear to be a reasonable loss
        function to start out with
        
        MSE takes an optional weight parameter
        which can be used to individually weight
        outputs when calculating loss. Will likely be
        used when we start concentrating on specific
        frequency bins  
    '''
    if mode != learn.ModeKeys.INFER:
        loss = tf.losses.mean_squared_error(
            labels=actual_next_state,
            predictions=predicted_next_state
            )
    
    # Configure the Training Op (for TRAIN mode)
    '''
    Can choose from the following optimizers
    
    OPTIMIZER_CLS_NAMES = {
        "Adagrad": train.AdagradOptimizer,
        "Adam": train.AdamOptimizer,
        "Ftrl": train.FtrlOptimizer,
        "Momentum": train.MomentumOptimizer,
        "RMSProp": train.RMSPropOptimizer,
        "SGD": train.GradientDescentOptimizer,
        }
    '''
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=tf.contrib.framework.get_global_step(),
            learning_rate=0.001,
            optimizer="SGD")
    
    # Generate Predictions
    predictions = { "next_state": predicted_next_state }
    
    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
            mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def main(unused_argv):
    # Load training and eval data
    mnist = learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    
    # Create the Estimator
    mnist_classifier = learn.Estimator(
            model_fn=cnn_model_EEG, model_dir="/tmp/mnist_convnet_model")
    
    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)
    
    sess = tf.Session()
    writer = tf.summary.FileWriter(".\\Logs\\",sess.graph)
    # Train the model
'''
    mnist_classifier.fit(
        x=train_data,
        y=train_labels,
        batch_size=100,
        steps=20000,
        monitors=[logging_hook])
    
    # Configure the accuracy metric for evaluation
    metrics = {
        "accuracy":
        learn.MetricSpec(
        metric_fn=tf.metrics.accuracy, prediction_key="classes"),
        }
    
    # Evaluate the model and print results
    eval_results = mnist_classifier.evaluate(
            x=eval_data, y=eval_labels, metrics=metrics)
    print(eval_results)
'''

if __name__ == "__main__":
    tf.app.run()
