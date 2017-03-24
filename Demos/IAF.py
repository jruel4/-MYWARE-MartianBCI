# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 15:28:38 2017

@author: marzipan
"""

from MartianBCI.Pipeline import Pipeline
from MartianBCI.Blocks.block_FIR import block_fir_filter

# Init and run pipeline
pipeline = Pipeline(buf_len_secs=0.004, chan_sel=0, sample_update_interval=1)
pipeline.select_source()
pipeline.add_block(block_fir_filter) 
pipeline.select_output()
pipeline.run()

# Get real data stream and visualize in vispy





