# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 12:54:48 2017

@author: marzipan
"""

from MartianBCI.Pipeline import Pipeline
from MartianBCI.Blocks.block_Inc_Spectrogram import block_inc_spectrogram

# Init and run pipeline
pipeline = Pipeline(buf_len_secs=0.016, chan_sel=list(range(8)), sample_update_interval=4)
pipeline.select_source()
pipeline.add_block(block_inc_spectrogram, nperseg=256, num_ch=8, window='hanning')
pipeline.select_output()
pipeline.run()





