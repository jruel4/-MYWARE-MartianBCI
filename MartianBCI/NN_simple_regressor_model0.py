# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

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
        Input ->
        Dense ->
        Dropout ->
        Dense (Output)
'''

# HYPERPARAMETERS
# dense0_size
# dropout0_rate

def simple_regressor_model0(features, labels, mode, params, config):
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
    
    dense0_size = params['simple_regressor_model0_dense0_size']
    dropout0_rate = params['simple_regressor_model0_dropout0_rate']

    print(specgram)
    print(audio)
    print(actual_next_state)

    """Model function for CNN."""
    
    '''
    We assume that our input is a spectrogram of shape: [batch, freqs, timesteps, nchan]
    And real values (!)
    '''

    # Input Reshaping
    # Flatten the spectrogram and audio inputs
    # Input Tensor Shape: [batch_size, freqs, timesteps, nchan], [batch_size, freqs, amps]
    # Output Tensor Shape: [batch_size, s_freqs * timesteps * nchan], [batch_size * a_freqs * amps]
    specgram_flat = tf.contrib.layers.flatten(specgram)
    audio_flat = tf.contrib.layers.flatten(audio)

    # Create Input Layer
    # Join audio and spectrogram data to use as input to NN
    # Input Tensor Shapes: [batch_size, s_freqs * timesteps * nchan], [batch_size * a_freqs * amps]
    # Output Tensor Shape: [batch_size, s_freqs * timesteps * nchan + batch_size * a_freqs * amps]
    # NOTE: Denoting the input size as [batch_size, features] from here-on-out
    input_layer = tf.concat([specgram_flat, audio_flat], axis=1)
    print(input_layer)

    # Dense Layer 1
    # Densely connected layer with 8192 neurons
    # Input Tensor Shape: [batch_size, features]
    # Output Tensor Shape: [batch_size, 8192]
    dense0 = tf.layers.dense(inputs=input_layer,
                            units=dense0_size,
                            activation=tf.nn.relu)
    
    
    print(dense0)
    # Dropout 1
    # Add dropout operation; 0.5 probability that element will be kept
    dropout0 = tf.layers.dropout(inputs=dense0,
                                rate=dropout0_rate,
                                training=mode == learn.ModeKeys.TRAIN)
    
    print(dropout0)
    # JCR
    # Add a final, fully connected layer which will represent our output state
    output_layer = tf.layers.dense(inputs=dropout0,
                                   units=nchan * spectrogram_freqs,
                                   activation=tf.nn.relu)
    
    print(output_layer)

    predicted_next_state = tf.reshape(output_layer, [-1,spectrogram_freqs, nchan])
    
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
    

    dirry="C:\\Users\\marzipan\\workspace\\MartianBCI\\MartianBCI\\Logs\\NNTesting_Regressor2\\"
    writer = tf.summary.FileWriter(dirry)
    
    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = dense0.name
    embedding_config.metadata_path = dirry + 'meta.tsv'
    # Specify the width and height of a single thumbnail.
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)
    
    # Generate Predictions
    predictions = { "predicted_next_state": predicted_next_state }    
    
    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
            mode=mode, predictions=predictions, loss=loss, train_op=train_op)