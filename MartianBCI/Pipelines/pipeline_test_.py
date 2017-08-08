# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 23:39:34 2017

@author: marzipan
"""

# Pipeline blocks
from MartianBCI.Pipeline import Pipeline
from MartianBCI.Blocks.Block_LSL import Block_LSL
from MartianBCI.Blocks.block_fir import block_fir
from MartianBCI.Blocks.block_periodogram import block_periodogram
from MartianBCI.Blocks.block_periodogram_raw import block_periodogram_raw
from MartianBCI.Blocks.block_reshape import block_reshape

from MartianBCI.Utils.lsl_utils import create_fake_eeg
from scipy import signal
import numpy as np

#create_fake_eeg()


#GLOBALS
G_FS = 250
G_SIGLEN = 100
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

# Filter
fir_block_0 = pipeline.add_block(
        _BLOCK=block_fir,
        _PARENT_UID="RAW",
        _CHANNELS=8)

# Add LSL output
#==============================================================================
# block_LSL0 = pipeline.add_block(
#         _BLOCK=Block_LSL,
#         _PARENT_UID=fir_block_0,
#         _parent_output_key='filtered',
#         stream_name='filtered_ts_0',
#         stream_type='EEG')
# 
#==============================================================================
block_LSL1 = pipeline.add_block(
        _BLOCK=Block_LSL,
        _PARENT_UID=fir_block_0,
        _parent_output_key='default',
        stream_name='raw_ts_0',
        stream_type='EEG')


# Spectrogram
periodo_block1 = pipeline.add_block(
        _BLOCK=block_periodogram,
        _PARENT_UID=fir_block_0,
        _NCHAN=G_NCHAN,
        _BUFLEN=G_SIGLEN,
        nfft=G_nFFT,
        window=G_Window)


# Flatten spectrogram
periodo_block_flat1 = pipeline.add_block(
        _BLOCK=block_reshape,
        _PARENT_UID=periodo_block1,
        _INPUT_SHAPE=G_PeriodoShape,
        _OUTPUT_SHAPE=[-1]) #make 1D

# Add LSL output
block_LSL2 = pipeline.add_block(
        _BLOCK=Block_LSL,
        _PARENT_UID=periodo_block_flat1,
        _parent_output_key='default',
        stream_name="FILT_" + G_PeriodoName,
        stream_type='PROC')

# Run
pipeline.run()