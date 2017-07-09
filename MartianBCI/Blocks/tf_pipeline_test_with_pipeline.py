# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from MartianBCI.Pipeline import Pipeline
from MartianBCI.Pipeline_TF import Pipeline_TF
from MartianBCI.Blocks.Block_LSL import Block_LSL
from MartianBCI.Blocks.Block_TF import Block_TF
from MartianBCI.Blocks.block_Test import test_block
from MartianBCI.Blocks.tf_block_test import tf_block_test
from MartianBCI.Blocks.tf_block_band_powers import tf_block_band_powers
from MartianBCI.Blocks.tf_block_IAF import tf_block_IAF
from MartianBCI.Blocks.tf_block_adapter_io import tf_block_adapter_io
from MartianBCI.Blocks.tf_block_fir import tf_block_fir
from MartianBCI.Blocks.tf_block_slice import tf_block_slice
from MartianBCI.Blocks.tf_block_convert2uv import tf_block_convert2uv
from MartianBCI.Utils.lsl_utils import create_multi_ch_test_source
from MartianBCI.Utils.lsl_utils import create_fake_eeg

fir_coeffs = '7-14_BP_FIR_BLACKMANN.npy'
#b,a = np.load('C:\\Conda\\MartianBCI\\Demos\\RL_Utils\\Filters\\' + fir_coeffs)
b,a = np.load('C:\\Users\\marzipan\\workspace\\MartianBCI\\Demos\\RL_Utils\\Filters\\' + fir_coeffs)
G_NCHAN=8
G_NSAMPLES=1000

# Create test data
#create_multi_ch_test_source(freqs=[9,8,10,12,30,29,40,45])
#create_fake_eeg()

tf.reset_default_graph()

pipeline_tf = Pipeline_TF([G_NCHAN,G_NSAMPLES])
#FIR Bandpass
fir_idx = pipeline_tf.add_block(_BLOCK=tf_block_fir, _PARENT_UID="RAW",
                                _NCHAN=G_NCHAN, _COEFFS=b)
#IAF Detection
iaf_idx = pipeline_tf.add_block(_BLOCK=tf_block_IAF, _PARENT_UID=fir_idx,
                                _NCHAN=G_NCHAN)
#Only take most single output of data from FIR (with slight delay)
slice_idx = pipeline_tf.add_block(_BLOCK=tf_block_slice, _PARENT_UID=fir_idx,
                                  _BEGIN=[0,G_NSAMPLES - 100], _SIZE=[-1,1], _OUTPUT_LEN=G_NCHAN)

#Add TF Pipeline outputs
pipeline_tf.make_block_output(_BLOCK_UID=iaf_idx, _BLOCK_OUTPUT_KEY="iaf")
pipeline_tf.make_block_output(_BLOCK_UID=slice_idx, _BLOCK_OUTPUT_KEY="slice")

# Init and run actual pipeline
pipeline = Pipeline(_BUF_LEN_SECS=0.004, _CHAN_SEL=list(range(8)), _SAMPLE_UPDATE_INTERVAL=1)
pipeline.select_source()
block_adapter0 = pipeline.add_block(_BLOCK=tf_block_adapter_io,
                                    _PARENT_UID="RAW",
                                    _PIPE_TF=pipeline_tf,
                                    _INPUT_SHAPE=[G_NCHAN,G_NSAMPLES],
                                    _WRITE_GRAPH=True,
                                    _WRITE_SUMMARIES=True,
                                    _WRITE_METADATA_N_STEPS=500)

block_LSL0 = pipeline.add_block(_BLOCK=Block_LSL, _PARENT_UID=block_adapter0, _parent_output_key='iaf', stream_name='IAF')
block_LSL1 = pipeline.add_block(_BLOCK=Block_LSL, _PARENT_UID=block_adapter0, _parent_output_key='slice', stream_name='SLICE_FIR')
pipeline.run()
