# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 00:43:04 2017

@author: marzipan
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 00:21:49 2017

@author: marzipan
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 23:39:34 2017

@author: marzipan
"""

import numpy as np
import time

# Pipeline blocks
from MartianBCI.Pipeline import Pipeline
from MartianBCI.Blocks.Block_LSL import Block_LSL
from MartianBCI.Blocks.block_morlet import block_morlet
from MartianBCI.Blocks.block_noise_reject import block_noise_reject
from MartianBCI.Blocks.block_reshape import block_reshape


#GLOBALS
G_FS = 250
G_SIGLEN = 500
G_NCHAN = 8

#Morlet frequencies
G_Freqs = np.arange(8,30,0.1)
G_FreqsFWHM = [(f,2) for f in G_Freqs]

# Periodogram shape variables
G_MorletOutputShape = (G_NCHAN, len(G_Freqs))

# Stream name
G_MorletStreamName = "Morlet_" + str(G_MorletOutputShape[0]) + 'x' + str(G_MorletOutputShape[1]) + '_' + str(np.prod(G_MorletOutputShape)) + time.strftime("_%H:%M:%S")


pipeline = Pipeline(_BUF_LEN_SECS=0.004, _CHAN_SEL=list(range(G_NCHAN)), _SAMPLE_UPDATE_INTERVAL=1)
pipeline.select_source()

# Noise reject
noise_reject0 = pipeline.add_block(
        _BLOCK=block_noise_reject,
        _PARENT_UID="RAW")

# Morlet
morlet0 = pipeline.add_block(
        _BLOCK=block_morlet,
        _PARENT_UID=noise_reject0,
        f_fwhm = G_FreqsFWHM)

# Flatten morlet
morlet_RS0 = pipeline.add_block(
        _BLOCK=block_reshape,
        _PARENT_UID=morlet0,
        _INPUT_SHAPE=G_MorletOutputShape,
        _OUTPUT_SHAPE=[-1]) #make 1D

# Add LSL output
block_LSL0 = pipeline.add_block(
        _BLOCK=Block_LSL,
        _PARENT_UID=morlet_RS0,
        _parent_output_key='default',
        stream_name=G_MorletStreamName,
        stream_type='MORLET')

# Run
pipeline.run()