# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

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


# Pipeline blocks
from MartianBCI.Pipeline import Pipeline
from MartianBCI.Blocks.Block_LSL import Block_LSL
from MartianBCI.Blocks.block_spectrogram import block_spectrogram
from MartianBCI.Blocks.block_reshape import block_reshape

# Neural net
from MartianBCI.NN_CNNDropoutDense import cnn_model_EEG_0
from MartianBCI.NN_simple_regressor_model0 import simple_regressor_model0

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils as TFUtils

#GLOBALS
G_FS = 250
G_SIGLEN = 250
G_NCHAN = 8
G_logs = 'C:\\Users\\marzipan\\workspace\\MartianBCI\\MartianBCI\\Logs\\'
G_logdir = G_logs + 'NNNoModel_Custom\\'

# Spectrogram parameters
G_nPerSeg = 125
G_nOverlap = 0
G_nFFT = 125

# Spectrogram shape variables
G_InputShape = [G_NCHAN, G_SIGLEN]
# Creates np zeros in the shape of the input, calls spectro w/ above params, slices off Sxx and returns shape
G_SpectrogramShape = signal.spectrogram(np.zeros(G_InputShape), fs=G_FS, nperseg=G_nPerSeg, noverlap=G_nOverlap, nfft=G_nFFT, axis=1)[2].shape
G_SpectrogramLen = G_SpectrogramShape[2]
G_SpectrogramFreqs = G_SpectrogramShape[1]

# Regressor Model 0 Hyperparameters
G_dense0_size = 8192
G_dropout0_rate = 0.5


### PREPROCESSING PIPELINE
def eeg_preprocessing():
    pipeline = Pipeline(_BUF_LEN_SECS=0.004, _CHAN_SEL=list(range(G_NCHAN)), _SAMPLE_UPDATE_INTERVAL=1)
    pipeline.select_source()
    
    # Spectrogram
    spectro_block0 = pipeline.add_block(
            _BLOCK=block_spectrogram,
            _PARENT_UID="RAW",
            _INPUT_SHAPE=G_InputShape,
            fs=G_FS,
            nperseg=G_nPerSeg,
            noverlap=G_nOverlap,
            nfft=G_nFFT)
    
    # Flatten spectrogram
    spectro_block_flat0 = pipeline.add_block(
            _BLOCK=block_reshape,
            _PARENT_UID=spectro_block0,
            _INPUT_SHAPE=G_SpectrogramShape,
            _OUTPUT_SHAPE=[-1]) #make 1D
    
    # Add LSL output
    block_LSL0 = pipeline.add_block(
            _BLOCK=Block_LSL,
            _PARENT_UID=spectro_block_flat0,
            _parent_output_key='reshape',
            stream_name='SpectroFlat',
            stream_type='PROC')

    # Run
    pipeline.run()
    return pipeline
    


# CONFIGURE INPUT LSL STREAMS (TO NN)
def setup_lsl_stream():
    ## ACQUIRE LSL
    streams = resolve_stream()
    for i,s in enumerate(streams):
        print(i,"\t",s.name(),"\t",s.source_id())
    stream_id = input("NN LSL, input desired stream id: ")
    inlet = StreamInlet(streams[int(stream_id)])
    return inlet

def setup_lsl_eeg_stream():
    return setup_lsl_stream()

def setup_lsl_audio_stream():
    return setup_lsl_stream()




# LSL INPUT LOOP
def lsl_acquisition_thread(inlets, queues):
    global G_RunThreadEvent
    delay = 1.0/G_FS

    # Try using G_RunThreadEvent
    try:
        while G_RunThreadEvent.is_set():
            # Pull samples from all inlets and add to correspondign queue
            for idx in range(len(inlets)):
                data,timestamp = inlets[idx].pull_sample()
                try:
                    queues[idx].put(data)
                except Full:
                    print("Queue full")
    
            # Sleep after we've ran through all inlets
            time.sleep(delay)

    except NameError:
        print("LSL Acquisition Thread should be called from inside of run handler but thread event doesn't exist.")
        return




# NN LIVE INPUT FN
def input_fn_live(local_queue,test=False):
    global q_len
    global queue_input
    eeg_feat = list()
    label = list()
    
    #new_spect should be pulling in an entirely new spectrogram each time (super inefficient but for now...)
    q_len.append(local_queue.qsize())
    for i in range(np.minimum(local_queue.qsize(), 500)):
        if test:
            t,f,Sxx_in = signal.spectrogram(
                    np.asarray([[np.sin(2*np.pi*(x/250)*20) for x in range(G_SIGLEN)]]*G_NCHAN), #10Hz
                    fs=G_FS,
                    nperseg=G_nPerSeg,
                    noverlap=G_nOverlap,
                    nfft=G_nFFT,
                    axis=1)
            Sxx_in = Sxx_in.ravel()
        else:
            # Will raise Empty if empty; should never happen, should ALWAYS be checked before being called
            Sxx_in = np.asarray(local_queue.get(block=True,timeout=0.004))
    
        # Debugging
#        q_len.append(local_queue.qsize())
    
        # Unravel the spectrogram (LSL only outputs in 1D)
        Sxx_unravel = np.reshape(Sxx_in, [G_NCHAN, G_SpectrogramFreqs, G_SpectrogramLen])
    
        # Transform to match NN input, from [nchan, freqs, timesteps] to [freqs, timesteps, nchan]
        Sxx = np.transpose(Sxx_unravel, [1,2,0])
    
        # NN Expects absolute value and float32
        Sxx = np.abs(Sxx).astype(np.float32)
    
        # Features are timesteps 0:N-1, label is N
        eeg_feat.append(Sxx[:,0:-1,:])
        label.append(Sxx[:,-1,:])
    
    # Temporarily generate bullshit audio data
    audio = np.asarray([[0,0,0,0]]*len(eeg_feat)).astype(np.float32)
    return {'specgram':np.asarray(eeg_feat),
            'audio': audio}, {'next_state':np.asarray(label)}










def main(unused_argv):
    global pipeline
    global q_len
    q_len = list()    
    global G_RunThreadEvent
    global G_PipelineHandle

    # Create our mutex
    G_RunThreadEvent =  Event()

    # Set run flag
    G_RunThreadEvent.set()
        
    # Ininitialize the input preprocessing pipeline
    G_PipelineHandle = eeg_preprocessing()

    # Setup queues
#    AudioInputQueue = Queue() # Local fifo queue, usage: .put(), .get()
    EEGInputQueue = Queue() # Local fifo queue, usage: .put(), .get()

    # Setup inlets
#    AudioInlet = setup_lsl_audio_stream()
    EEGInlet = setup_lsl_eeg_stream()

    # Create inputs for acquisition thread
    inlets = [EEGInlet]
    queues = [EEGInputQueue]
#    inlets = [EEGInlet, AudioInlet]
#    queues = [EEGInputQueue, AudioInputQueue]

    # Initialize acquisition thread
    acquisition_thread = Thread(target=lambda: lsl_acquisition_thread(inlets, queues))
    acquisition_thread.start()
    
    
    
    
    
    
    
    
    
    
    # -*- coding: utf-8 -*-





    # Unpack the input features and labels
    actual_next_state = tf.placeholder(tf.float32, shape=[1,G_SpectrogramFreqs, G_NCHAN])
    specgram = tf.placeholder(tf.float32, shape=[1, G_SpectrogramFreqs, G_SpectrogramLen - 1, G_NCHAN])
    audio = tf.placeholder(tf.float32, shape=[1, 4])

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
                            units=G_dense0_size,
                            activation=tf.nn.relu)
    
    
    print(dense0)
    # Dropout 1
    # Add dropout operation; 0.5 probability that element will be kept
    dropout0 = tf.layers.dropout(inputs=dense0,
                                rate=G_dropout0_rate,
                                training=True)
    
    print(dropout0)
    # Add a final, fully connected layer which will represent our output state
    output_layer = tf.layers.dense(inputs=dropout0,
                                   units=G_NCHAN * G_SpectrogramFreqs,
                                   activation=tf.nn.relu)
    
    print(output_layer)

    predicted_next_state = tf.reshape(output_layer, [-1,G_SpectrogramFreqs, G_NCHAN])
    
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

    loss = tf.losses.mean_squared_error(
        labels=actual_next_state,
        predictions=predicted_next_state
        )
    
    
    # Train Op    
    tf.summary.scalar('loss', loss)
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    
    #
    writer = tf.summary.FileWriter(G_logdir)
    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = dense0.name
    embedding_config.metadata_path = G_logdir + 'emb_mappings.tsv'
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)
    
#    with tf.Graph().as_default():
    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(G_logdir, sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)

    # Start the training loop.
    for step in range(500):
      start_time = time.time()

      feat,lbl = input_fn_live(EEGInputQueue,False)
      feed_dict = {
              specgram:feat['specgram'],
              audio:feat['audio'],
              actual_next_state:lbl['next_state']
                   }

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % 50 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 100 == 0:
        print("Saving checkpoint")
        checkpoint_file = G_logdir + 'model.ckpt'
        saver.save(sess, checkpoint_file, global_step=step)

        
    
    '''

    Neural net needs to accept reasonable oscillator parameters to try
    Neural net will output set of state features used to calculate reward
    Neural net module will output should be optimal freq / amp for each oscillator (entrainment) based on best features
    LSL Output from neural net

    '''
    # We can use this to save the entire model for later training
    #mnist_classifier.export_savedmodel('.\\Logs\\ModelGraph\\',input_fn=generate_eeg_data_INFUN)

    G_PipelineHandle.stop()
#    acquisition_thread.stop()
    G_RunThreadEvent.clear()
    return

if __name__ == "__main__":
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()