# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from MartianBCI.Pipeline import Pipeline
from MartianBCI.Blocks.tf_block_fir import tf_block_fir
from MartianBCI.Blocks.tf_block_convert2uv import tf_block_convert2uv
from MartianBCI.Blocks.tf_block_IAF import tf_block_IAF
from MartianBCI.Blocks.tf_block_band_powers import tf_block_band_powers
from MartianBCI.Blocks.tf_block_adapter_io import tf_block_adapter_io
from MartianBCI.Blocks.tf_block_adapter_io import tf_block_adapter_add
from MartianBCI.Utils.lsl_utils import create_multi_ch_test_source



# Create test data
create_multi_ch_test_source(freqs=[9,8,10,12,30,29,40,45])

fir_coeffs = '7-14_BP_FIR_BLACKMANN.npy'
#b,a = np.load('C:\\Conda\\MartianBCI\\Demos\\RL_Utils\\Filters\\' + fir_coeffs)
b,a = np.load('C:\\Users\\marzipan\\workspace\\MartianBCI\\Demos\\RL_Utils\\Filters\\' + fir_coeffs)

tf.reset_default_graph()

NCHAN=8
SIGLEN=1000
blocks = [
        tf_block_adapter_add(tf_block_fir, NCHAN, b.astype(np.float32)),
        tf_block_adapter_add(tf_block_convert2uv, NCHAN, 24),
        tf_block_adapter_add(tf_block_IAF, NCHAN)
        ]



# Init and run pipeline
pipeline = Pipeline(buf_len_secs=0.004, chan_sel=list(range(NCHAN)), sample_update_interval=1)
pipeline.select_source()
pipeline.add_block(tf_block_adapter_io, blocks, _LEN=SIGLEN, _NCHAN=NCHAN)
pipeline.select_output()
pipeline.run()