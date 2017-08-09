# -*- coding: utf-8 -*-
"""
Created on Tue Aug 08 19:54:30 2017

@author: marzipan
"""

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
from MartianBCI.Blocks.block_reshape import block_reshape
from MartianBCI.Blocks.block_noise_reject import block_noise_reject
from MartianBCI.Blocks.block_moving_average import block_moving_average
from MartianBCI.Blocks.block_normalize_periodogram import block_normalize_periodogram
from MartianBCI.Blocks.block_lambda import block_lambda

#GLOBALS
G_FS = 250
G_SIGLEN = 250
G_NCHAN = 8

#Morlet frequencies
G_Freqs = np.arange(1,30,1)
#G_Freqs = np.asarray([10,15,20,25])
G_FreqsFWHM = [(f,4) for f in G_Freqs]

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

#==============================================================================
# conv2 = pipeline.add_block(
#     _BLOCK=block_lambda,
#     _PARENT_UID=morlet0,
#     _LAMBDA_FUNC = lambda x: x * 1e6,
#     _OUTPUT_SHAPE=[8,30] )
#==============================================================================

# Log baseline
morlet_baseline0 = pipeline.add_block(
        _BLOCK=block_normalize_periodogram,
        _PARENT_UID=morlet0,
        _num_freqs=G_MorletOutputShape[1],
        _num_chan=G_MorletOutputShape[0],
        continuous_baseline=False)

#==============================================================================
# # Moving average
# ma_block0 = pipeline.add_block(
#         _BLOCK=block_moving_average,
#         _PARENT_UID=morlet_baseline0,
#         _NCHAN=G_MorletOutputShape[0],
#         _NFREQS =G_MorletOutputShape[1],
#         _MA_LEN=250 )
#==============================================================================

# Flatten morlet
morlet_RS0 = pipeline.add_block(
        _BLOCK=block_reshape,
        _PARENT_UID=morlet_baseline0,
        _INPUT_SHAPE=G_MorletOutputShape,
        _OUTPUT_SHAPE=[-1]) #make 1D

# Add LSL output 0
block_LSL0 = pipeline.add_block(
        _BLOCK=Block_LSL,
        _PARENT_UID=morlet_RS0,
        _parent_output_key='default',
        stream_name=G_MorletStreamName,
        stream_type='MORLET')
        


def bslice(v):
    return lambda x, idx=v: x[idx]



main = [ pipeline.add_block( _BLOCK=block_lambda, _PARENT_UID=morlet_RS0, _LAMBDA_FUNC = bslice(5 + x*29), _OUTPUT_SHAPE=1 ) for x in range(8) ] +  [ pipeline.add_block( _BLOCK=block_lambda, _PARENT_UID=morlet_RS0, _LAMBDA_FUNC = bslice(8 + x*30), _OUTPUT_SHAPE=1 ) for x in range(2) ]
energy = [ pipeline.add_block( _BLOCK=block_lambda, _PARENT_UID=morlet_RS0, _LAMBDA_FUNC = bslice(9 + x*29), _OUTPUT_SHAPE=1 ) for x in range(8) ] +  [ pipeline.add_block( _BLOCK=block_lambda, _PARENT_UID=morlet_RS0, _LAMBDA_FUNC = bslice(8 + x*30), _OUTPUT_SHAPE=1 ) for x in range(2) ]
fatigue = [ pipeline.add_block( _BLOCK=block_lambda, _PARENT_UID=morlet_RS0, _LAMBDA_FUNC = bslice(13 + x*29), _OUTPUT_SHAPE=1 ) for x in range(8) ] +  [ pipeline.add_block( _BLOCK=block_lambda, _PARENT_UID=morlet_RS0, _LAMBDA_FUNC = bslice(8 + x*30), _OUTPUT_SHAPE=1 ) for x in range(2) ]
stress = [ pipeline.add_block( _BLOCK=block_lambda, _PARENT_UID=morlet_RS0, _LAMBDA_FUNC = bslice(17 + x*29), _OUTPUT_SHAPE=1 ) for x in range(8) ] +  [ pipeline.add_block( _BLOCK=block_lambda, _PARENT_UID=morlet_RS0, _LAMBDA_FUNC = bslice(8 + x*30), _OUTPUT_SHAPE=1 ) for x in range(2) ]
cognitiveload = [ pipeline.add_block( _BLOCK=block_lambda, _PARENT_UID=morlet_RS0, _LAMBDA_FUNC = bslice(21 + x*29), _OUTPUT_SHAPE=1 ) for x in range(8) ] +  [ pipeline.add_block( _BLOCK=block_lambda, _PARENT_UID=morlet_RS0, _LAMBDA_FUNC = bslice(8 + x*30), _OUTPUT_SHAPE=1 ) for x in range(2) ]

emot = [ pipeline.add_block( _BLOCK=block_lambda, _PARENT_UID=morlet_RS0, _LAMBDA_FUNC = bslice(x+29*5), _OUTPUT_SHAPE=1 ) for x in range(10,20)]

# Add LSL outlet blocks
lsl_outputs = [
pipeline.add_block(
        _BLOCK=Block_LSL,
        _PARENT_UID=main[i],
        _parent_output_key='default',
        stream_name="Main" + str(i),
        stream_type='PROC')
for i in range(10)
] + [
pipeline.add_block(
        _BLOCK=Block_LSL,
        _PARENT_UID=energy[i],
        _parent_output_key='default',
        stream_name="Energy" + str(i),
        stream_type='PROC')
for i in range(10)
] + [
pipeline.add_block(
        _BLOCK=Block_LSL,
        _PARENT_UID=fatigue[i],
        _parent_output_key='default',
        stream_name="Fatigue" + str(i),
        stream_type='PROC')
for i in range(10)
] + [
pipeline.add_block(
        _BLOCK=Block_LSL,
        _PARENT_UID=stress[i],
        _parent_output_key='default',
        stream_name="Stress" + str(i),
        stream_type='PROC')
for i in range(10)
] + [
pipeline.add_block(
        _BLOCK=Block_LSL,
        _PARENT_UID=cognitiveload[i],
        _parent_output_key='default',
        stream_name="CognitiveLoad" + str(i),
        stream_type='PROC')
for i in range(10)
] + [
pipeline.add_block(
        _BLOCK=Block_LSL,
        _PARENT_UID=emot[i],
        _parent_output_key='default',
        stream_name="EmotionalValence" + str(i),
        stream_type='PROC')
for i in range(10)
]

# Run
pipeline.run()

# Record baseline: 
#H_blinefunc = pipeline.get_block_handle(morlet_baseline0)
#H_blinefunc.record_baseline(duration_seconds=30)