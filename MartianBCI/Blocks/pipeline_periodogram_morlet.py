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

from scipy import signal
import numpy as np

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

# Periodogram parameters
G_nFFT = 500
G_Window = signal.tukey(G_SIGLEN,0.25)

# Periodogram shape variables
G_InputShape = [G_NCHAN, G_SIGLEN]

# Creates np zeros in the shape of the input, call fft and get shape
G_PeriodoShape = np.abs(np.fft.fft(np.zeros(G_InputShape), n=G_nFFT)).shape

# Generate Periodogram stream name (include info about periodo)
G_PeriodoName='PeriodoFlat_' + str(G_PeriodoShape[0]) + 'x' + str(G_PeriodoShape[1]) + '_' + str(np.prod(G_PeriodoShape))

pipeline = Pipeline(_BUF_LEN_SECS=0.004, _CHAN_SEL=list(range(G_NCHAN)), _SAMPLE_UPDATE_INTERVAL=1)
pipeline.select_source()


# Morlet
periodo_block0 = pipeline.add_block(
        _BLOCK=block_periodogram,
        _PARENT_UID="RAW",
        _INPUT_SHAPE=G_InputShape,
        nfft=G_nFFT,
        window=G_Window)

# Spectrogram
periodo_block0 = pipeline.add_block(
        _BLOCK=block_periodogram,
        _PARENT_UID="RAW",
        _INPUT_SHAPE=G_InputShape,
        nfft=G_nFFT,
        window=G_Window)

# Flatten spectrogram
periodo_block_flat0 = pipeline.add_block(
        _BLOCK=block_reshape,
        _PARENT_UID=periodo_block0,
        _INPUT_SHAPE=G_PeriodoShape,
        _OUTPUT_SHAPE=[-1]) #make 1D

# Add LSL output
block_LSL0 = pipeline.add_block(
        _BLOCK=Block_LSL,
        _PARENT_UID=periodo_block_flat0,
        _parent_output_key='reshape',
        stream_name=G_PeriodoName,
        stream_type='PROC')

# Run
pipeline.run()