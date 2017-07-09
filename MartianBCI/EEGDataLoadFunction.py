# -*- coding: utf-8 -*-

from MartianBCI.Pipeline_TF import Pipeline_TF
from Demos.RL_Utils.Helpers.LoadXDF import get_raw_data
import numpy as np
import tensorflow as tf
from scipy import signal



from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils as TFUtils


def generate_eeg_data(step=0,_TF_OUTPUT=True):

    # Load XDF
    global xdf_raw
    global sess
    global output_data
    global output_labels

    try:
        xdf_raw
    except NameError:
        xdf_raw = get_raw_data('C:\\Users\\marzipan\\workspace\\MartianBCI\\Demos\\RL_Utils\\Helpers\\Recordings\\JCR_IAF_06-22-17.xdf')[0]
    xdf_raw_np = np.asarray( np.transpose( xdf_raw ) )
    
    try:
        b
    except NameError:
        fir_coeffs = '7-14_BP_FIR_BLACKMANN.npy'
        #b,a = np.load('C:\\Conda\\MartianBCI\\Demos\\RL_Utils\\Filters\\' + fir_coeffs)
        b,a = np.load('C:\\Users\\marzipan\\workspace\\MartianBCI\\Demos\\RL_Utils\\Filters\\' + fir_coeffs)
    
    
    #Pipeline
    pipeline_tf = Pipeline_TF([G_NCHAN,G_NSAMPLES])
    
    #Filter block
    flt_UID = pipeline_tf.add_block(
            _BLOCK=tf_block_fir,
            _PARENT_UID="RAW",
            _NCHAN=G_NCHAN,
            _COEFFS=b)

    #Spectrogram block
    specgram_UID = pipeline_tf.add_block(
            _BLOCK=tf_block_specgram,
            _PARENT_UID=flt_UID,
            _NCHAN=G_NCHAN,
            _INPUT_LEN=G_NSAMPLES,
            _WINDOW_LEN=G_WINDOW_LEN,
            _WINDOW_STRIDE=G_WINDOW_STRIDE)

    #Build pipeline graph
    pipeline_tf.make_block_output(specgram_UID,'specgram')
    pp_graph_in, pp_graph_out = pipeline_tf.build_main_graph()
    #specgram = pp_graph_out['data']['specgram']
    
    
    sess = tf.Session()
    tf.summary.FileWriter(".\\Logs\\generate_eeg_data",sess.graph)
    
    output_data = list()
    output_labels = list()
    for idx in range(100):
        p = sess.run(pp_graph_out,
                     {pp_graph_in: xdf_raw_np[:, (2500 + G_WINDOW_LEN * idx ): (2500 + G_NSAMPLES + G_WINDOW_LEN * idx)]}
                     )
        output_data.append(p['data']['specgram'][1:-1, :,:])
        output_labels.append(p['data']['specgram'][-1:, :,:])

    if _TF_OUTPUT:
        
        feat = tf.constant(np.asarray(output_data))
        label = tf.constant(np.asarray(output_labels))
    else:
        feat = np.asarray(output_data)
        label = np.asarray(output_labels)

    sess.close()   
    return {'specgram':feat, 'audio': tf.constant(0)},{'next_state':label}


def generate_eeg_data_default():
    f,l = generate_eeg_data(_TF_OUTPUT=True)
    ifo = TFUtils.InputFnOps(
            features=f,
            labels=l,
            default_inputs=f)
    return ifo