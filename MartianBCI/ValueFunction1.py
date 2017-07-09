# -*- coding: utf-8 -*-

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
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils as TFUtils



## BUG
## Doesn't work when G_BATCH_SIZE < nFFT

#GLOBALS
G_FS = 250
G_UpdateInterval = 1.0 / 50.0
G_SIGLEN = 500
G_NCHAN = 8
G_BATCH_SIZE = 1
G_logs = 'C:\\Users\\marzipan\\workspace\\MartianBCI\\MartianBCI\\Logs\\'
G_logdir = G_logs + 'ValueFunction_MaybeWorking5_RandomKernelInit_3\\'

# Spectrogram parameters
G_nPerSeg = 250
G_nOverlap = 0
G_nFFT = 500

# Spectrogram shape variables
G_InputShape = [G_NCHAN, G_SIGLEN]
# Creates np zeros in the shape of the input, calls spectro w/ above params, slices off Sxx and returns shape

G_Freqs, _, G_SpectrogramShape = signal.spectrogram(np.zeros([G_BATCH_SIZE] + G_InputShape), fs=G_FS, nperseg=G_nPerSeg, noverlap=G_nOverlap, nfft=G_nFFT)
G_SpectrogramShape = G_SpectrogramShape.shape
G_SpectrogramLen = G_SpectrogramShape[-1]
G_SpectrogramFreqs = G_SpectrogramShape[-2]

# Regressor Model 0 Hyperparameters
G_dense0_size = 5000
G_dropout0_rate = 0.2

# Super hacky, get rid of this
G_optimal_profile_mask = np.tile(np.asarray([[int(x == 10)]* G_NCHAN for x in G_Freqs]),[G_BATCH_SIZE,1,1])


G_f1 = np.linspace(0.5,30,50)
G_f2 = np.linspace(0.5,30,50)
G_a1 = np.linspace(0,1,10)
G_a2 = np.linspace(0,1,10)

G_valid_audio = np.asarray(
        [[f1,f2,a1,a2] for
        f1 in G_f1 for
        f2 in G_f2 for
        a1 in G_a1 for
        a2 in G_a2]
        )

'''
Accepts 4 inputs corresponding to the current audio parameters and outputs all neighbors within a radius of "r" 
'''
def valid_next_audio_steps(f1,f2,a1,a2,r=2):
    try:
        f1idx = int(np.where(G_f1 == f1)[0])
        f2idx = int(np.where(G_f2 == f2)[0])
        a1idx = int(np.where(G_a1 == a1)[0])
        a2idx = int(np.where(G_a2 == a2)[0])
    except TypeError:
        raise TypeError("Error, input audio steps appears not to be valid / recognised. Alternatively, the audio arrays may have been created incorrectly. Receieved input: ", f1,f2,a1,a2)
        

    
    return np.asarray(
        [[v_f1,v_f2,v_a1,v_a2] for
        v_f1 in G_f1[max([0, f1idx - r]) : min([len(G_f1), f1idx + r + 1])] for
        v_f2 in G_f2[max([0, f2idx - r]) : min([len(G_f2), f2idx + r + 1])] for
        v_a1 in G_a1[max([0, a1idx - r]) : min([len(G_a1), a1idx + r + 1])] for
        v_a2 in G_a2[max([0, a2idx - r]) : min([len(G_a2), a2idx + r + 1])]]
        )

### PREPROCESSING PIPELINE
def eeg_preprocessing():
    pipeline = Pipeline(_BUF_LEN_SECS=(.004), _CHAN_SEL=list(range(G_NCHAN)), _SAMPLE_UPDATE_INTERVAL=int(1))
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
            _INPUT_SHAPE= G_SpectrogramShape[1:4],
            _OUTPUT_SHAPE=[-1]) #make 1D
    
    # Add LSL output
    block_LSL0 = pipeline.add_block(
            _BLOCK=Block_LSL,
            _PARENT_UID=spectro_block_flat0,
            _parent_output_key='reshape',
            stream_name='SpectroFlat_MASTER__',
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

    # Try using G_RunThreadEvent
    try:
        while G_RunThreadEvent.is_set():
            # Pull samples from all inlets and add to correspondign queue
            for idx in range(len(inlets)):
                data,timestamp = inlets[idx].pull_sample(timeout=0)
                if data != None:
                    try:
                        queues[idx].put(data)
                    except Full:
                        print("Queue full")
    except NameError:
        print("LSL Acquisition Thread should be called from inside of run handler but thread event doesn't exist.")
        return



G_TestSig = np.asarray(
                [[[ 1.2*np.sin(2*np.pi*(t/G_FS)*(10+freq)) + 10.0*np.sin(2*np.pi*(t/G_FS)) + 0.7*np.random.rand()
                for t in range(G_SIGLEN)]
                for freq in range(G_NCHAN)]
                for batch in range(G_BATCH_SIZE)]
    )


dur0=list()
dur1=list()
dur2=list()
    
# NN LIVE INPUT FN
def input_fn_live(local_queue, previous_expected_value, f1,f2,a1,a2, timeout=(G_UpdateInterval), test=False):
    global q_len
    global dur0, dur1, dur2

    eeg_feat = list()
    label = list()
    
    # Logging
    q_len.append(local_queue.qsize())
    
    # Pull in next spectrograms (or, if in test mode, generate them)
    Sxx_in = list()
    if test:
        # Declare this global so we don't have to constantly re-generate it
        global G_TestSig

        # Time the spectrograms
        a=time.time()
        t,f,Sxx_in = signal.spectrogram(
                G_TestSig,
                fs=G_FS,
                nperseg=G_nPerSeg,
                noverlap=G_nOverlap,
                nfft=G_nFFT)
        print("INFN Test: Spectro ", time.time() - a)
        a=time.time()
        Sxx_in = Sxx_in.ravel()
        print("INFN: Ravel ", time.time() - a)
    else:
        # Wait until the queue is full
        a = time.time()
        while local_queue.qsize() < G_BATCH_SIZE:
            pass

        for i in range(G_BATCH_SIZE):
            # Will raise Empty if empty; should never happen, should ALWAYS be checked before being called
            a=time.time()
            guy=local_queue.get(block=True,timeout=timeout)
            dur0.append(time.time()-a)

            a=time.time()
            npguy=np.asarray(guy)
            dur1.append(time.time()-a)

            a=time.time()
            Sxx_in.append(npguy)
            dur2.append(time.time()-a)


    # Unravel the spectrogram (LSL only outputs in 1D)
    a=time.time()
    Sxx_unravel = np.reshape(Sxx_in, [G_BATCH_SIZE, G_NCHAN, G_SpectrogramFreqs, G_SpectrogramLen])
#    print("INFN: Unravel ", time.time() - a)
    # Transform to match NN input, from [batch_size, nchan, freqs, timesteps] to [batch_size, freqs, timesteps, nchan]
    a=time.time()
    Sxx = np.transpose(Sxx_unravel, [0,2,3,1])
#    print("INFN: Transpose ", time.time() - a)
    # NN Expects absolute value and float32
    Sxx = np.abs(Sxx).astype(np.float32)

    # Features are timesteps 0:N-1, label is N
    eeg_feat = Sxx[:,:,0:-1,:]
#    print(eeg_feat.shape, " EE Feat Shape")
#    print(previous_expected_value.shape, " Prev Expected Val Shape")
    previous_expected_value = np.reshape(previous_expected_value, [-1,1])
#    print(previous_expected_value.shape, " Prev Expected Val Shape New")
    label = Sxx[:,:,-1,:]
    
    # Calculate Euclidean distance from current position to the optimal position
    value = np.linalg.norm(G_optimal_profile_mask * (100 - label), axis=(1,2))
    value = np.expand_dims(value,1)
    
    if False and ([f1,f2,a1,a2] == G_valid_audio[0]).all():
        print("Got it!")
        # This should be learned quickly!
        value = value * 0

    audio = np.asarray([[f1,f2,a1,a2]]*len(eeg_feat)).astype(np.float32)

    # Build output dictionary
    out = {'specgram':eeg_feat,
            'audio': audio,
            'previous_expected_value':previous_expected_value,
            'current_reward':value}
    return out


def main(unused_argv):
    global pipeline
    global q_len
    q_len = list()    
    global G_RunThreadEvent
    global G_PipelineHandle
    global EEGInputQueue

    f1=G_f1[1]
    f2=G_f2[1]
    a1=G_a1[1]
    a2=G_a2[1]

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




    # Unpack the input features and labels
#    actual_next_state = tf.placeholder(tf.float32)
#    specgram = tf.placeholder(tf.float32)
#    audio = tf.placeholder(tf.float32)
    prev_predicted_value = tf.placeholder(tf.float32, shape=[None, 1])
    current_reward = tf.placeholder(tf.float32, shape=[None,1])
    specgram = tf.placeholder(tf.float32, shape=[None, G_SpectrogramFreqs, G_SpectrogramLen - 1, G_NCHAN])
    audio = tf.placeholder(tf.float32, shape=[None, 4])

    # Input Reshaping
    # Flatten the spectrogram and audio inputs
    # Input Tensor Shape: [batch_size, freqs, timesteps, nchan], [batch_size, freqs, amps]
    # Output Tensor Shape: [batch_size, s_freqs * timesteps * nchan], [batch_size * a_freqs * amps]
    specgram_flat = tf.contrib.layers.flatten(specgram)
    audio_flat = tf.contrib.layers.flatten(audio)

    print(current_reward)
    print(specgram)
    print(audio)
    print(specgram_flat)
    print(audio_flat)
    
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
                            kernel_initializer=tf.random_uniform_initializer(),
#                            bias_initializer=tf.random_uniform_initializer(),
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
                                   units=1,
                                   activation=tf.nn.relu)
    
    print(output_layer)

    # What we predict the next state will be
    next_predicted_value = output_layer
    
    print(next_predicted_value)
    
    # Calculate Loss (for both TRAIN and EVAL modes)
    '''
    JCR NOTE:
        MSE would appear to be a reasonable loss
        function to start out with
    '''

    gamma_decay = tf.constant(0.2, name='GammaDecay')
    loss = tf.losses.mean_squared_error(
        labels=(current_reward + gamma_decay * next_predicted_value),
        predictions=prev_predicted_value
        )
    
    
    # Train Op    
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('act', current_reward[0,0])
    tf.summary.scalar('prev_predicted', prev_predicted_value[0,0])
    tf.summary.scalar('next_predicted', next_predicted_value[0,0])
    tf.summary.image('specgram',specgram[:,:,:,0:1])
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.AdamOptimizer(0.05)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    
    # Write new visualization
    writer = tf.summary.FileWriter(G_logdir)
    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    dense_emb = config.embeddings.add()
    dense_emb.tensor_name = 'dense/kernel'
    dense_emb.metadata_path = G_logdir + 'emb_mappings.tsv'
    dense_emb1 = config.embeddings.add()
    dense_emb1.tensor_name = 'dense/kernel/Adam'
    dense_emb1.metadata_path = G_logdir + 'emb_mappings.tsv'
    dense_emb2 = config.embeddings.add()
    dense_emb2.tensor_name = 'dense/kernel/Adam_1'
    dense_emb2.metadata_path = G_logdir + 'emb_mappings.tsv'
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)
    
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

    # START MAIN EVAL LOOP
    global batch_feats
    global batch_lbls
    global batch_shapes
    global durations
    global best_audio
    global next_audio
    best_audio = list()
    durations = list()
    batch_shapes = list()
    prev_predicted_value_lcl = np.zeros([1,1])
    for step in range(250):
        start=time.time()
        batch_inputs = list()

        a = time.time()
        # Only eval every second
        for i in range(1):
            inputs = input_fn_live(EEGInputQueue, prev_predicted_value_lcl, f1,f2,a1,a2,test=True)
            batch_inputs.append(inputs)

#        print("Acquire data: ", time.time() - a)
        batch_shapes.append(np.shape(batch_inputs))

        '''
        global feed_dict_batch
        feed_dict_batch = {
            specgram:np.concatenate([x['specgram'] for x in batch_inputs]),
            audio:np.concatenate([x['audio'] for x in batch_inputs]),
            prev_predicted_value:np.concatenate(x['previous_expected_value'] for x in batch_inputs),
            current_reward:np.concatenate([x['current_reward'] for x in batch_inputs])
            }
        '''

        # Evaluate - Calculate both A0 and best Q(S0,A0)
        s=time.time()
        next_audio = valid_next_audio_steps(f1,f2,a1,a2,r=5)
        feed_dict_eval = {
            specgram:np.tile(inputs['specgram'][0],[len(next_audio),1,1,1]),
            audio:next_audio,
# I don't think you need the placeholder if it's not used
#            actual_value:np.tile(lbl['next_value'][0],[len(next_audio),1])
#            prev_predicted_value:inputs['previous_expected_value'],
#            current_reward:inputs['current_reward']
            }
        arg_val = sess.run(next_predicted_value, feed_dict_eval)
        
        if (step + 1) % 50 == 0:
            print("Evaluation time taken: ", time.time() - s)
            print("Best Audio: ", next_audio[np.argmin(arg_val)])

        argmin_idx = np.argmin(arg_val)
        f1,f2,a1,a2 = next_audio[argmin_idx]
        
        # This temporarily stores the predicted value to use as the "previously predicted value" in the next iteration
        prev_predicted_value_lcl = arg_val[argmin_idx]
        
        
        # Setup the feed dict for the loss function
        feed_dict = {
            specgram:inputs['specgram'],
            audio:np.asarray([[f1,f2,a1,a2]]),
            prev_predicted_value:inputs['previous_expected_value'],
            current_reward:inputs['current_reward']
            }
        
        
        # Logging
        best_audio.append([f1,f2,a1,a2])
        durations.append(time.time() - start)
        
        # Train / Update Model
        start_time = time.time()
        _, loss_value = sess.run([train_op, loss],
                                 feed_dict=feed_dict)

        train_duration = time.time() - start_time

        # Write the summaries and print an overview fairly often.
        a=time.time()
        if step % 100 == 0:
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, train_duration))
            summaries = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summaries, step)
            summary_writer.flush()
#        print("Summary writing: ", time.time() - a)

        # Save a checkpoint periodically.
        a=time.time()
        if (step + 1) % 200 == 0:
            print("Saving checkpoint")
            checkpoint_file = G_logdir + 'model.ckpt'
            saver.save(sess, checkpoint_file, global_step=step)
#        print("Checkpoint writing: ", time.time() - a)

    # End of run loop

    G_PipelineHandle.stop()
#    acquisition_thread.stop()
    G_RunThreadEvent.clear()
    return

if __name__ == "__main__":
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()