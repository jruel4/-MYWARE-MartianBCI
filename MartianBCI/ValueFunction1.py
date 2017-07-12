# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf
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

## BUG
## Doesn't work when G_BATCH_SIZE < nFFT

#GLOBALS
G_FS = 250
G_UpdateInterval = 1.0 / 50.0
G_SIGLEN = 500
G_NCHAN = 8
G_BATCH_SIZE = 1
G_logs = 'C:\\Users\\marzipan\\workspace\\MartianBCI\\MartianBCI\\Logs\\'
G_logdir = G_logs + 'ValueFunction_MaybeWorking5_RandomKernelInit_7\\'

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
G_dense0_size = 100
G_dense1_size = 100
G_dropout0_rate = 0.2
G_dropout1_rate = 0.2


# Super hacky, get rid of this
G_optimal_profile_mask = np.tile(np.asarray([[int(x == 10)]* G_NCHAN for x in G_Freqs]),[G_BATCH_SIZE,1,1])


#G_f1 = np.linspace(0.5,30,50)
#G_f2 = np.linspace(0.5,30,50)

G_f1 = np.linspace(0,100,50)
G_f2 = np.linspace(0,100,50)
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



        
# protocol
# [ begin_freq, end_freq, percent_total_reward, increase/decrease ] 

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

n_len=5
s=0
func_tst = np.asarray([[x,x**2] for x in np.linspace(0,5,n_len)])
def input_fn_live_tst(local_queue, previous_expected_value, f1,f2,a1,a2, timeout=(G_UpdateInterval), test=False):
    global func_tst
    global s
    spect=np.zeros(G_SpectrogramShape)
    spect=np.asarray([np.transpose(spect, [3,2,0,1])[0,:,:,:]])
    
    s = (s+1) % n_len
    audio=np.asarray([[ func_tst[s,0], 0, 0, 0 ]])
    reward = np.asarray([[ func_tst[s,1] ]])
    out = {'specgram':spect,
            'audio': audio,
            'previous_expected_value':np.asarray([[100.0]]),
            'current_reward':reward}
    return out




# NN LIVE INPUT FN
def input_fn_live(local_queue, previous_expected_value, f1,f2,a1,a2, timeout=(G_UpdateInterval)):
    global q_len
    global dur0, dur1, dur2

    eeg_feat = list()
    label = list()
    
    # Logging (input queue length)
    q_len.append(local_queue.qsize())
    
    # Pull in next spectrograms (or, if in test mode, generate them)
    Sxx_in = list()

    # Wait until the queue is full
    while local_queue.qsize() < G_BATCH_SIZE: pass

    for i in range(G_BATCH_SIZE):
        # Will raise Empty if empty; should never happen, should ALWAYS be checked before being called
        Sxx_in.append(np.asarray(local_queue.get(block=True,timeout=timeout)))

    # Unravel the spectrogram (LSL only outputs in 1D)
    Sxx_unravel = np.reshape(Sxx_in, [G_BATCH_SIZE, G_NCHAN, G_SpectrogramFreqs, G_SpectrogramLen])

    # Transform to match NN input from:
    #   [batch_size, nchan, freqs, timesteps] to
    #   [batch_size, freqs, timesteps, nchan]
    # Also, NN Expects absolute value and float32
    Sxx = np.abs( np.transpose(Sxx_unravel, [0,2,3,1]) ).astype(np.float32)

    
    # Features are timesteps 0:N-1, label is N
    eeg_feat = Sxx[:,:,0:-1,:]
    previous_expected_value = np.reshape(previous_expected_value, [-1,1])
    label = Sxx[:,:,-1,:]
    
    # Calculate Euclidean distance from current position to the optimal position
    value = np.asarray([[100]]*len(eeg_feat))
#    value = np.linalg.norm(G_optimal_profile_mask * (100 - label), axis=(1,2))
#    value = np.expand_dims(value,1)

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
    value_pred_sneg1 = tf.placeholder(tf.float32, shape=[None, 1],name="PredictedValueS0")
    reward_s0_a0 = tf.placeholder(tf.float32, shape=[None,1])
    s0_specgram = tf.placeholder(tf.float32, shape=[None, G_SpectrogramFreqs, G_SpectrogramLen - 1, G_NCHAN])
    s0_audio = tf.placeholder(tf.float32, shape=[None, 4])
    print(reward_s0_a0)
    print(s0_specgram)
    print(s0_audio)

    # Input Reshaping
    # Flatten the spectrogram and audio inputs
    s0_specgram_flat = tf.contrib.layers.flatten(s0_specgram)
    s0_audio_flat = tf.contrib.layers.flatten(s0_audio)
    print(s0_specgram_flat)
    print(s0_audio_flat)
    
    # Create Input Layer
    # Join audio and spectrogram data to use as input layer
    input_layer = tf.concat([s0_specgram_flat, s0_audio_flat], axis=1)
    print(input_layer)

    # Dense Layer 1
    # Densely connected layer with 8192 neurons
    dense0 = tf.layers.dense(
                inputs=input_layer,
                units=G_dense0_size,
                kernel_initializer=tf.random_uniform_initializer(),
                bias_initializer=tf.random_uniform_initializer(),
                activation=tf.sigmoid)    
    print(dense0)

    # Dropout 1
    # Add dropout operation; 0.5 probability that element will be kept
    dropout0 = tf.layers.dropout(inputs=dense0,
                                rate=G_dropout0_rate,
                                training=False)
    print(dropout0)


    # Dense Layer 1
    # Densely connected layer with 8192 neurons
    dense1 = tf.layers.dense(
                inputs=input_layer,
                units=G_dense1_size,
#                kernel_initializer=tf.random_uniform_initializer(),
#                bias_initializer=tf.random_uniform_initializer(),
                activation=tf.sigmoid)    
    print(dense1)

    # Dropout 2
    # Add dropout operation; 0.5 probability that element will be kept
    dropout1 = tf.layers.dropout(inputs=dense1,
                                rate=G_dropout1_rate,
                                training=False)
    print(dropout1)


    # Add a final, fully connected layer which will represent our output state
    output_layer = tf.layers.dense(inputs=dropout1,
                                   units=1,
                                   activation=None)
    print(output_layer)

    # V(S0,A0)
    value_pred_s0 = output_layer
        
    # Calculate Loss (for both TRAIN and EVAL modes)
    '''
    JCR NOTE:
        MSE would appear to be a reasonable loss
        function to start out with
    '''
    gamma_decay = tf.constant(0.2, name='GammaDecay')
    loss = tf.losses.mean_squared_error(
        labels=(reward_s0_a0 + gamma_decay * value_pred_s0),
        predictions=value_pred_sneg1
        )
    
    # Define Summaries
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('reward', reward_s0_a0[0,0])
    tf.summary.scalar('prev_predicted', value_pred_sneg1[0,0])
    tf.summary.scalar('next_predicted', value_pred_s0[0,0])
#    tf.summary.image('specgram',s0_specgram[0,:,:,0:1])

    # Write new visualization
    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    d0_kern = config.embeddings.add()
    d0_kern.tensor_name = 'dense/kernel'
    d0_kern.metadata_path = G_logdir + 'emb_mappings.tsv'    

    # TRAINING OPS
    
    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(5e-3)

    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    
    
    # TF INIT
    
    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    global sess
    sess = tf.Session()

    # Create the summary writer
    summary_writer = tf.summary.FileWriter(G_logdir, sess.graph)
    summary = tf.summary.merge_all()
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(summary_writer, config)

    # Run the Op to initialize the variables.
    sess.run(init)


    # START MAIN EVAL LOOP
    
    global batch_feats
    global batch_lbls
    global batch_shapes
    global durations
    global best_audio
    global next_audio
    global d0_act
    d0_act = list()
    global o0_act
    o0_act = list()
    global do0_act
    do0_act = list()
    global l0_act
    l0_act = list()
    global bw_act
    bw_act = list()

    best_audio = list()
    durations = list()
    batch_shapes = list()
    value_pred_s0_lcl = np.zeros([1,1])

    spect=np.ones(G_SpectrogramShape)
    spect=np.asarray([np.transpose(spect, [3,2,0,1])[0,:,:,:]])

    for step in range(50):
        start=time.time()
        batch_inputs = list()

        # Acquire new data
        a = time.time()
        inputs = input_fn_live(EEGInputQueue, value_pred_s0_lcl, f1,f2,a1,a2)
        batch_inputs.append(inputs)
        batch_shapes.append(np.shape(batch_inputs))

        # Get next valid audio
        next_audio = valid_next_audio_steps(f1,f2,a1,a2,r=2)
        feed_dict_greedy_select = {
            s0_specgram     :   np.tile(inputs['specgram'][0],[len(next_audio),1,1,1]),
            s0_audio        :   next_audio
            }
        
        # Generate greedy Q(S0,A0)
        possible_action_values = sess.run(value_pred_s0, feed_dict_greedy_select)
        
        # Get A0 from agrmin(Q(S0,A0))
        argmin_idx = np.argmin(possible_action_values)
        f1,f2,a1,a2 = next_audio[argmin_idx]
        
        # This temporarily stores the predicted value to use as the "previously predicted value" in the next iteration
        value_pred_s0_lcl = possible_action_values[argmin_idx]
        l0_act.append(value_pred_s0_lcl)
        
        if (step + 1) % 25 == 0:
            print("Evaluation time taken: ", time.time() - s)
            print("Best Audio: ", next_audio[np.argmin(possible_action_values)])
            best_audio.append([f1,f2,a1,a2])

        # Setup the feed dict for the loss function
        feed_dict_training = {
            s0_specgram         :   inputs['specgram'],
            s0_audio            :   inputs['audio'],
            value_pred_sneg1    :   inputs['previous_expected_value'],
            reward_s0_a0        :   inputs['current_reward'],
            }        
        
        # Weights
        b_w = dict()
        for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            b_w.update({i.name: sess.run(i)})
        bw_act.append(b_w)
        
        
        # Loss checking
        d0,do0,o0,l0 = sess.run([dense0,dropout0,output_layer,loss], feed_dict=feed_dict_training)
        d0_act.append(d0)
        do0_act.append(do0)
        o0_act.append(o0)
        l0_act.append(l0)
        
        
        # Logging
        
        durations.append(time.time() - start)
        
        # Train / Update Model
        start_time = time.time()
        _, loss_value = sess.run([train_op, loss],
                                 feed_dict=feed_dict_training)

        train_duration = time.time() - start_time

        # Write the summaries and print an overview fairly often.
        if step % 10 == 0:
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, train_duration))
            summaries = sess.run(summary, feed_dict=feed_dict_training)
            summary_writer.add_summary(summaries, step)
            summary_writer.flush()
    
        # Save a checkpoint periodically.
        if (step + 1) % 50 == 0:
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