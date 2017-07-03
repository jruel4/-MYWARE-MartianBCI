# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 05:55:07 2017

@author: marzipan
"""

from MartianBCI.Pipeline import Pipeline
from MartianBCI.Blocks.block_Inc_Spectrogram import block_inc_spectrogram
from MartianBCI.Blocks.block_freq_bin import block_freq_bin
from MartianBCI.Utils.lsl_utils import create_multi_ch_test_source

# Create test data
#create_multi_ch_test_source(freqs=[9,8,10,12,30,29,40,45])

# Init and run pipeline
pipeline = Pipeline(buf_len_secs=0.016, chan_sel=list(range(8)), sample_update_interval=4)
pipeline.select_source()
pipeline.add_block(block_inc_spectrogram, nperseg=256, num_ch=8, window='hanning')
pipeline.add_block(block_freq_bin)
pipeline.select_output()
pipeline.run()

