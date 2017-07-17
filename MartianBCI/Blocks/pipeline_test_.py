# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 23:39:34 2017

@author: marzipan
"""

# Pipeline blocks
from MartianBCI.Pipeline import Pipeline
from MartianBCI.Blocks.Block_LSL import Block_LSL
from MartianBCI.Blocks.block_fir import block_fir

from MartianBCI.Utils.lsl_utils import create_fake_eeg
from scipy import signal
import numpy as np

create_fake_eeg()

#GLOBALS
G_FS = 250
G_SIGLEN = 1000
G_NCHAN = 8

pipeline = Pipeline(_BUF_LEN_SECS=0.004, _CHAN_SEL=list(range(G_NCHAN)), _SAMPLE_UPDATE_INTERVAL=1)
pipeline.select_source()

# Filter
fir_block_0 = pipeline.add_block(
        _BLOCK=block_fir,
        _PARENT_UID="RAW")

# Add LSL output
block_LSL0 = pipeline.add_block(
        _BLOCK=Block_LSL,
        _PARENT_UID=fir_block_0,
        _parent_output_key='filtered',
        stream_name='filtered_ts_0',
        stream_type='EEG')

# Run
pipeline.run()









