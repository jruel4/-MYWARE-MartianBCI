# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib



'''
G_FS=250
G_NCHAN=16
G_NSAMPLES=5000

#PARAMETERS
G_WINDOW_LEN = 200
G_WINDOW_STRIDE = 100
G_SPECGRAM_LEN = (G_NSAMPLES - G_WINDOW_LEN) // G_WINDOW_STRIDE + 1

cnn0_num_kernels = 32
cnn0_kernel_height =    2
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
'''

'''

CNN Model EEG:
    Inputs:
        features - dict containing:
            ['spectro']:
                spectrogram block for all channels; assumed formate is time_points x nchan x window_len
            ['audio']:
                audio data (NOT IMPLEMENTED)
        labels - dict containing:
            'next_state':
                next EEG state; format is (flat([#chan x freq])), i.e (c0,f0), (c0,f1), ... (cN, fN-1), (cN, fN)
        mode - train/infer/eval (keys from learn.ModeKeys)
    Outputs:
        ModelFnOps for use with tf.contrib.learn.Estimator

    Structure:
        CNN ->
        MAXPOOL ->
        CNN ->
        [MAXPOOL, AUDIO] ->
        Densely connected ->
        Output layer
'''

def cnn_model_EEG_0(features, labels, mode, params, config):
    # Unpack the input features and labels
    actual_next_state = labels['next_state']
    specgram = features['specgram']
    audio = features['audio']    

    # Unpack the input parameters
    fs = params['SamplingFrequency']
    nchan = params['NumberOfChannels']
    siglen = params['InputLength']
    spectrogram_len = params['SpectrogramTimeSteps'] - 1 # most recent spectro slice is used as label
    spectrogram_freqs = params['SpectrogramFreqSteps']
    

    """Model function for CNN."""
    
    '''
    We assume that our input is a spectrogram of shape: [batch, freqs, timesteps, nchan]
    And real values (!)
    '''

    print(params)

    # Input Reshaping
    # We split input into seperate tensors for each
    # channel b/c channels are not directly
    # related, at least in current model
    #
    # Input Tensor Shape: [batch_size, freqs, timesteps, nchan]
    # Output Tensor Shape: n_ch * [batch_size, freqs, timesteps, 1]
    input_layer = tf.split(specgram, nchan, axis=3)
    print(input_layer)
    
    '''
    Note that CNN input has shape:
        1 x time_steps x window_len x channels
        
    Importantly, spectrograms are:
        time_steps x window_len
    
    Typical values would be something like:
        10 x 200
    
    Keep this in mind when setting kernel parameters.
            
    '''
    
    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: n_ch * [batch_size, freqs, timesteps, 1]
    # Output Tensor Shape: n_ch * [batch_size, freqs, timesteps, 32]
    conv1 = [
            tf.layers.conv2d(
                inputs=x,
                filters=32,
                kernel_size=[2, 2],
        #        stride=[1,1],
                padding="SAME",
                activation=tf.nn.relu)
            for x in input_layer]
    print("C1:",conv1[0])
    
    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: n_ch * [batch_size, freqs, timesteps, 32]
    # Output Tensor Shape: n_ch * [batch_size, freqs/2, timesteps/2, 32]
    pool1 = [
            tf.layers.max_pooling2d(
                    inputs=x,
                    pool_size=[2, 2],
                    strides=2)
            for x in conv1 ]
    print("P1:",pool1[0])

    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: n_ch * [batch_size, freqs/2, timesteps/2, 32]
    # Output Tensor Shape: n_ch * [batch_size, freqs/2, timesteps/2, 64]
    conv2 = [
            tf.layers.conv2d(
                inputs=x,
                filters=4,
                kernel_size=[2, 2],
                padding="same",
                activation=tf.nn.relu)
            for x in pool1]
    print("C2",conv2[0])

    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: n_ch * [batch_size, freqs/2, timesteps/2, 64]
    # Output Tensor Shape: n_ch * [batch_size, freqs/4, timesteps/4, 64]
    pool2 = [
            tf.layers.max_pooling2d(
                    inputs=x,
                    pool_size=[2, 2],
                    strides=2)
            for x in conv2]
    print("P2:",pool2[0])

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: n_ch * [batch_size, freqs/2, timesteps/2, 64]
    # Output Tensor Shape: [batch_size, freqs/4 * timesteps/4 * 64 * n_ch]
    pool2_joined = tf.stack(pool2,-1)
    print(pool2_joined)
    pool2_flat = tf.reshape(pool2_joined, [-1, int((spectrogram_freqs//4) * (spectrogram_len//4) * 4 * nchan)])
    print(pool2_flat)
    
    # JCR
    # Dense Layer
    # Densely connected layer with 65536 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=nchan * spectrogram_freqs, activation=tf.nn.relu)
    
    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
            inputs=dense, rate=0.1, training=mode == learn.ModeKeys.TRAIN)
    
    
    # JCR
    # Add a final, fully connected layer which will represent our output state
    predicted_next_state_tmp = tf.layers.dense(inputs=dropout,
                                           units=nchan * spectrogram_freqs,
                                           activation=tf.nn.relu)
    
    predicted_next_state = tf.reshape(predicted_next_state_tmp, [-1, spectrogram_freqs, nchan])
#    predicted_next_state = tf.expand_dims(predicted_next_state3d, 1)

    loss = None
    train_op = None
    
    # Calculate Loss (for both TRAIN and EVAL modes)
    '''
    JCR NOTE:
        MSE would appear to be a reasonable loss
        function to start out with
        
        MSE takes an optional weight parameter
        which can be used to individually weight
        outputs when calculating loss. Maybe use this
        when we start concentrating on specific
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
            learning_rate=0.1,
            optimizer="SGD")
    
    # Generate Predictions
    predictions = { "predicted_next_state": predicted_next_state }
    
    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
            mode=mode, predictions=predictions, loss=loss, train_op=train_op)