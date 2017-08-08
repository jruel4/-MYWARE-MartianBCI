# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 00:56:03 2017

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

# Pipeline blocks
from MartianBCI.Pipeline import Pipeline
from MartianBCI.Blocks.Block_LSL import Block_LSL
from MartianBCI.Blocks.block_periodogram import block_periodogram
from MartianBCI.Blocks.block_moving_average import block_moving_average
from MartianBCI.Blocks.block_reshape import block_reshape
from MartianBCI.Blocks.block_lambda import block_lambda

from scipy import signal
import numpy as np

#GLOBALS
G_FS = 250
G_SIGLEN = 250
G_NCHAN = 1

# Periodogram parameters
G_nFFT = 500
G_Window = signal.tukey(G_SIGLEN,0.5)

G_RemoveDC = True
G_RemoveMirror = True

# Periodogram shape variables
G_InputShape = [G_NCHAN, G_SIGLEN]

# Creates np zeros in the shape of the input, call fft and get shape
G_PeriodoShape = np.abs(np.fft.fft(np.zeros(G_InputShape), n=G_nFFT)).shape

G_PeriodoShape = (G_PeriodoShape[0], G_PeriodoShape[1] / (1 + int(G_RemoveMirror)) - int(G_RemoveDC))

# Generate Periodogram stream name (include info about periodo)
G_PeriodoName='Main'

pipeline = Pipeline(_BUF_LEN_SECS=0.004, _CHAN_SEL=list(range(G_NCHAN)), _SAMPLE_UPDATE_INTERVAL=1)
pipeline.select_source()

# Spectrogram
periodo_block0 = pipeline.add_block(
        _BLOCK=block_periodogram,
        _PARENT_UID="RAW",
        _NCHAN=G_NCHAN,
        _BUFLEN=G_SIGLEN,
        remove_dc=G_RemoveDC,
        remove_mirror=G_RemoveMirror,
        nfft=G_nFFT,
        window=G_Window)

ma_block0 = pipeline.add_block(
        _BLOCK=block_moving_average,
        _PARENT_UID=periodo_block0,
        _NCHAN=G_PeriodoShape[0],
        _NFREQS =G_PeriodoShape[1],
        _MA_LEN=50 )

# Flatten spectrogram
periodo_block_flat0 = pipeline.add_block(
        _BLOCK=block_reshape,
        _PARENT_UID=ma_block0,
        _INPUT_SHAPE=G_PeriodoShape,
        _OUTPUT_SHAPE=[-1]) #make 1D

# Slice one

freqs = np.linspace(0,125,(G_nFFT//2))[1:]
def getFreqIndex(f): return np.argmin(abs(freqs - f))
def sumFreq(x, f): return np.sum(x[getFreqIndex(f)::G_PeriodoShape[1]])

main =[
pipeline.add_block( _BLOCK=block_lambda, _PARENT_UID=periodo_block_flat0, _LAMBDA_FUNC = lambda x: sumFreq(x, 1), _OUTPUT_SHAPE=1 ),
pipeline.add_block( _BLOCK=block_lambda, _PARENT_UID=periodo_block_flat0, _LAMBDA_FUNC = lambda x: sumFreq(x, 3), _OUTPUT_SHAPE=1 ),
pipeline.add_block( _BLOCK=block_lambda, _PARENT_UID=periodo_block_flat0, _LAMBDA_FUNC = lambda x: sumFreq(x, 5), _OUTPUT_SHAPE=1 ),
pipeline.add_block( _BLOCK=block_lambda, _PARENT_UID=periodo_block_flat0, _LAMBDA_FUNC = lambda x: sumFreq(x, 8), _OUTPUT_SHAPE=1 ),
pipeline.add_block( _BLOCK=block_lambda, _PARENT_UID=periodo_block_flat0, _LAMBDA_FUNC = lambda x: sumFreq(x, 11), _OUTPUT_SHAPE=1 ),
pipeline.add_block( _BLOCK=block_lambda, _PARENT_UID=periodo_block_flat0, _LAMBDA_FUNC = lambda x: sumFreq(x, 14), _OUTPUT_SHAPE=1 ),
pipeline.add_block( _BLOCK=block_lambda, _PARENT_UID=periodo_block_flat0, _LAMBDA_FUNC = lambda x: sumFreq(x, 18), _OUTPUT_SHAPE=1 ),
pipeline.add_block( _BLOCK=block_lambda, _PARENT_UID=periodo_block_flat0, _LAMBDA_FUNC = lambda x: sumFreq(x, 22), _OUTPUT_SHAPE=1 ),
pipeline.add_block( _BLOCK=block_lambda, _PARENT_UID=periodo_block_flat0, _LAMBDA_FUNC = lambda x: sumFreq(x, 28), _OUTPUT_SHAPE=1 ),
pipeline.add_block( _BLOCK=block_lambda, _PARENT_UID=periodo_block_flat0, _LAMBDA_FUNC = lambda x: sumFreq(x, 35), _OUTPUT_SHAPE=1 )
]

energy =[
pipeline.add_block(
        _BLOCK=block_lambda,
        _PARENT_UID=periodo_block_flat0,
        _LAMBDA_FUNC = lambda x, idx=i: x[getFreqIndex(10+idx)],
        _OUTPUT_SHAPE=1
)
for i in range(10)]

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
]


# Run
pipeline.run()