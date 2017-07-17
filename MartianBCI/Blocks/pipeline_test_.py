# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 23:39:34 2017

@author: marzipan
"""

# Pipeline blocks
from MartianBCI.Pipeline import Pipeline
from MartianBCI.Blocks.Block_LSL import Block_LSL
from MartianBCI.Blocks.block_spectrogram import block_spectrogram
from MartianBCI.Blocks.block_reshape import block_reshape

from MartianBCI.Utils.lsl_utils import create_fake_eeg

from scipy import signal

import numpy as np

create_fake_eeg()

#GLOBALS
G_FS = 250
G_SIGLEN = 1000
G_NCHAN = 8

# Spectrogram parameters
G_nPerSeg = 125
G_nOverlap = 0
G_nFFT = 125

# Spectrogram shape variables
G_InputShape = [G_NCHAN, G_SIGLEN]
# Creates np zeros in the shape of the input, calls spectro w/ above params, slices off Sxx and returns shape
G_SpectrogramShape = signal.spectrogram(np.zeros(G_InputShape), fs=G_FS, nperseg=G_nPerSeg, noverlap=G_nOverlap, nfft=G_nFFT, axis=1)[2].shape

pipeline = Pipeline(_BUF_LEN_SECS=0.004, _CHAN_SEL=list(range(G_NCHAN)), _SAMPLE_UPDATE_INTERVAL=1)
pipeline.select_source()

# Spectrogram
spectro_block0 = pipeline.add_block(
        _BLOCK=block_spectrogram,
        _PARENT_UID="RAW",
        _INPUT_SHAPE=[G_NCHAN,G_SIGLEN],
        fs=G_FS,
        nperseg=G_nPerSeg,
        noverlap=G_nOverlap,
        nfft=G_nFFT)

# Flatten spectrogram
spectro_block_flat0 = pipeline.add_block(
        _BLOCK=block_reshape,
        _PARENT_UID=spectro_block0,
        _INPUT_SHAPE=G_SpectrogramShape,
        _OUTPUT_SHAPE=[-1]) #make 1D

# Add LSL output
block_LSL0 = pipeline.add_block(
        _BLOCK=Block_LSL,
        _PARENT_UID=spectro_block_flat0,
        _parent_output_key='reshape',
        stream_name='SpectroFlat',
        stream_type='PROC')

# Run
pipeline.run()