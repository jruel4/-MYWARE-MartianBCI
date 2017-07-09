# -*- coding: utf-8 -*-

'''
Useful for debugging
'''
from MartianBCI.Pipeline_TF import Pipeline_TF
from Demos.RL_Utils.Helpers.LoadXDF import get_raw_data
import numpy as np
import tensorflow as tf
from scipy import signal
import timeit
from Demos.RL_Utils.Helpers import Processing_TF as Utils
from MartianBCI.Blocks.tf_block_specgram import tf_block_specgram
from MartianBCI.Blocks.tf_block_fir import tf_block_fir

tf.reset_default_graph()

G_FS=250
G_NCHAN=16
G_NSAMPLES=500

fir_coeffs = '7-14_BP_FIR_BLACKMANN.npy'
#b,a = np.load('C:\\Conda\\MartianBCI\\Demos\\RL_Utils\\Filters\\' + fir_coeffs)
b,a = np.load('C:\\Users\\marzipan\\workspace\\MartianBCI\\Demos\\RL_Utils\\Filters\\' + fir_coeffs)

#If xdf_raw doesn't exist then load it
try:
    xdf_raw
except NameError:
    xdf_raw=get_raw_data('C:\\Users\\marzipan\\workspace\\MartianBCI\\Demos\\RL_Utils\\Helpers\\Recordings\\JCR_IAF_06-22-17.xdf')[0]

#Only use a slice of the input (full thing is like 55K x 16 values, a bit much...)
xdf_raw_slice=xdf_raw[2500:2500 + G_NSAMPLES]
tf_input = np.asarray(np.transpose(xdf_raw_slice))
print("Raw input shape: ", tf_input.shape, " corresponding to ", tf_input.shape[1]/250, " seconds of data.")

#PARAMETERS
G_WINDOW_LEN = 200
G_WINDOW_STRIDE = 5
G_SPECGRAM_LEN = (G_NSAMPLES - G_WINDOW_LEN) // G_WINDOW_STRIDE + 1

cnn0_kernel_height =    5
cnn0_kernel_width =     10
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
      (G_SPECGRAM_LEN / cnn0_kernel_stride) * (G_WINDOW_LEN / cnn0_kernel_stride))

#Setup a few variables for ease of readability
cnn0_kernel = [cnn0_kernel_height, cnn0_kernel_width]
cnn1_kernel = [cnn0_kernel_height, cnn0_kernel_width, 10]



#Pipeline
pipeline_tf = Pipeline_TF([G_NCHAN,G_NSAMPLES])

#Filter block
'''
flt_UID = pipeline_tf.add_block(
        _BLOCK=tf_block_fir,
        _PARENT_UID="RAW",
        _NCHAN=G_NCHAN,
        _COEFFS=b)
'''
#Spectrogram block
specgram_UID = pipeline_tf.add_block(
        _BLOCK=tf_block_specgram,
        _PARENT_UID="RAW",
        _NCHAN=G_NCHAN,
        _INPUT_LEN=G_NSAMPLES,
        _WINDOW_LEN=G_WINDOW_LEN,
        _WINDOW_STRIDE=G_WINDOW_STRIDE)

pipeline_tf.make_block_output(specgram_UID,'specgram')
pp_graph_in, pp_graph_out = pipeline_tf.build_main_graph()

#Prepare the spectrogram output for processing using CNNs (Batch x ImageWidth x ImageHeight x Channels)
specgram = pp_graph_out['data']['specgram']
specgram_4d=tf.expand_dims(specgram,0)
specgram_NWHC=tf.abs(tf.transpose(specgram_4d,[0,1,3,2]))


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
#conv2d_in_plane keeps channels seperate
#NOTEL in_plane does not support seperate stride sizes - might not be good
s4=tf.split(specgram_NWHC, G_NCHAN, 3)
conv1 = tf.contrib.layers.conv2d_in_plane(
    inputs=specgram_NWHC,
    kernel_size=cnn0_kernel,
    stride=cnn0_kernel_stride,
    padding="SAME",
    activation_fn=tf.nn.relu)


conv2 = tf.contrib.layers.conv2d(
    inputs=s4,
    num_outputs=5,
    kernel_size=cnn1_kernel,
    stride=cnn0_kernel_stride,
    padding="SAME",
    activation_fn=tf.nn.relu)


#dense = tf.layers.dense(inputs=s4, units=10, activation=tf.nn.relu)

sess = tf.Session()
writer = tf.summary.FileWriter(".\\Logs\\",sess.graph)

i,o =  {pp_graph_in:tf_input},pp_graph_out


sess.run(tf.global_variables_initializer())
#tc0=sess.run(conv1, i)
#tc1=sess.run(conv2, i)
tc2=sess.run(o, i)

#for idx in range(5):
#    p = sess.run(o,i)
run_options = tf.RunOptions(trace_level=tf.RunOptions.SOFTWARE_TRACE)
run_metadata = tf.RunMetadata()
p = sess.run(
        o,
        i,
        options=run_options,
        run_metadata=run_metadata)
writer.add_run_metadata(run_metadata, 'S: %d' % 1)

start_time = timeit.default_timer()
specto=signal.spectrogram(tf_input, fs=250, window=('tukey', 0.25), nperseg=200, noverlap=199,axis=1)
# code you want to evaluate
elapsed = timeit.default_timer() - start_time
print("Spec: ", elapsed)