# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 00:36:55 2017

@author: marzipan
"""

from MartianBCI.Pipeline import Pipeline
from MartianBCI.Blocks.block_spectrogram import block_spectrogram
from MartianBCI.Utils.lsl_utils import create_test_source
from MartianBCI.Feedback.spectrogram_mesh import visualize_spec_stream

# Create test data
create_test_source(freq=12, sps=250)

# Init and run pipeline
pipeline = Pipeline(buf_len_secs=5)
pipeline.select_source()
pipeline.add_block(block_spectrogram, nperseg=64)
pipeline.select_output()
pipeline.run()

# Execute Feedback 
visualize_spec_stream(pipeline)





