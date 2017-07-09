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

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils as TFUtils


tf.reset_default_graph()

#GLOBALS
G_FS = 250
G_SIGLEN = 250
G_NCHAN = 8
G_logs = 'C:\\Users\\marzipan\\workspace\\MartianBCI\\MartianBCI\\Logs\\'
G_logdir = G_logs + 'NNTesting_Regressor3\\'

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
simple_regressor_model0_dense0_size = 1000
simple_regressor_model0_dropout0_rate = 0.5

simple_regressor_model1_dense0_size = 8192
simple_regressor_model1_dropout0_rate = 0.5

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
        q_len.append(local_queue.qsize())
    
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
    return {'specgram':tf.constant(np.asarray(eeg_feat)),
            'audio': tf.constant(audio, name='audio_input')}, {'next_state':tf.constant(np.asarray(label))}


# Creates parameters for the NN, mostly just setting the correct shape of the tensors
def create_parameter_dict():
    parameters = {
     'SamplingFrequency':G_FS,
     'NumberOfChannels':G_NCHAN,
     'InputLength':G_SIGLEN,
     'SpectrogramTimeSteps':G_SpectrogramLen,
     'SpectrogramFreqSteps':G_SpectrogramFreqs,
     
     #Regressor Model 0
     'simple_regressor_model0_dense0_size':simple_regressor_model0_dense0_size,
     'simple_regressor_model0_dropout0_rate':simple_regressor_model0_dropout0_rate,
     
     # Regressor Model 1
     'logdir':G_logdir,
     'simple_regressor_model1_dense0_size':simple_regressor_model1_dense0_size,
     'simple_regressor_model1_dropout0_rate':simple_regressor_model1_dropout0_rate,
#     'optimal_profile':
     }
    return parameters


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
    
    # Create the Estimator
    simple_regressor = learn.Estimator(
            model_fn=simple_regressor_model0,
            model_dir=G_logdir,
            params=create_parameter_dict())
    
    # Set up logging for predictions
    tensors_to_log = {"probabilities": "audio_input"}
    logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=50)
    
    # Train the model
    for i in range(5):
        simple_regressor.partial_fit(
            input_fn=lambda: input_fn_live(EEGInputQueue,False),
            steps=1,
            monitors=[logging_hook])


    # Configure the accuracy metric for evaluation
    metrics = {
        "accuracy":
        learn.MetricSpec(
        metric_fn=tf.metrics.accuracy),
        }
    
    # Evaluate the model and print results
    eval_results = simple_regressor.evaluate(
            input_fn=lambda: input_fn_live(EEGInputQueue,False),
            metrics=metrics,
            steps=5)
    print(eval_results)
    
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