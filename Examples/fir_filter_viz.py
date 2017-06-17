# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 08:31:54 2017

@author: marzipan
"""


from MartianBCI.Pipeline import Pipeline
from MartianBCI.Blocks.block_FIR import block_fir_filter
from MartianBCI.Utils.lsl_utils import create_noisy_test_source

# Create test data
#create_noisy_test_source()

# Init and run pipeline
pipeline = Pipeline(buf_len_secs=0.004, chan_sel=list(range(8)), sample_update_interval=1)
pipeline.select_source()
pipeline.add_block(block_fir_filter) 
pipeline.select_output()
pipeline.run()

