# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 17:29:55 2017

@author: marzipan
"""

from MartianBCI.Pipeline import Pipeline
from MartianBCI.Blocks.block_Inc_Spectrogram import block_inc_spectrogram
from MartianBCI.Blocks.block_freq_bin import block_freq_bin
from MartianBCI.Blocks.block_moving_average import block_moving_average
from MartianBCI.Blocks.block_unity_normalize import block_unity_normalize
from MartianBCI.Utils.lsl_utils import tunable_alpha_signal

# Create test data
tuner = tunable_alpha_signal()

# Init and run pipeline
pipeline = Pipeline(buf_len_secs=0.016, chan_sel=list(range(8)), sample_update_interval=4)
pipeline.select_source()
pipeline.add_block(block_inc_spectrogram, nperseg=256, num_ch=8, window='hanning')
pipeline.add_block(block_freq_bin)
#pipeline.add_block(block_unity_normalize)
#pipeline.add_block(block_moving_average)
pipeline.select_output()
pipeline.run()

