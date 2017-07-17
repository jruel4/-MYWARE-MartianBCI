# -*- coding: utf-8 -*-

import numpy as np
from MartianBCI.Pipeline import Pipeline
from MartianBCI.Blocks.Block_LSL import Block_LSL
from MartianBCI.Blocks.block_Test import test_block
from MartianBCI.Utils.lsl_utils import create_multi_ch_test_source



# Create test data
create_multi_ch_test_source(freqs=[9,8,10,12,30,29,40,45])

# Init and run pipeline
pipeline = Pipeline(buf_len_secs=0.004, chan_sel=list(range(8)), sample_update_interval=1)
pipeline.select_source()
block_test = pipeline.add_block(test_block, "RAW", 1, 1)
block_LSL1 = pipeline.add_block(Block_LSL, block_test, 'data', 'T13')
pipeline.run()