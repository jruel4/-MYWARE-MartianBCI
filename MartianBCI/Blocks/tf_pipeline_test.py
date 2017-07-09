# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from MartianBCI.Pipeline import Pipeline
from MartianBCI.Pipeline_TF import Pipeline_TF
from MartianBCI.Blocks.Block_LSL import Block_LSL
from MartianBCI.Blocks.Block_TF import Block_TF
from MartianBCI.Blocks.block_Test import test_block
from MartianBCI.Blocks.tf_block_test import tf_block_test
from MartianBCI.Blocks.tf_block_band_powers import tf_block_band_powers
from MartianBCI.Blocks.tf_block_IAF import tf_block_IAF
from MartianBCI.Blocks.tf_block_adapter_io import tf_block_adapter_io
from MartianBCI.Blocks.tf_block_fir import tf_block_fir
from MartianBCI.Blocks.tf_block_slice import tf_block_slice
from MartianBCI.Blocks.tf_block_convert2uv import tf_block_convert2uv
from MartianBCI.Utils.lsl_utils import create_multi_ch_test_source

tf.reset_default_graph()

G_NCHAN=1

'''

BASIC TEST

'''

pipeline_tf = Pipeline_TF([G_NCHAN])
idx = pipeline_tf.add_block(tf_block_test, "RAW", 0,0)
pipeline_tf.make_block_output(idx,'x')
x,y = pipeline_tf.build_main_graph()
i0,o0 =  {x:np.asarray([100]*G_NCHAN)},y


'''

CONVERT 2 UV TEST

'''

pipeline_tf2 = Pipeline_TF([G_NCHAN,10])
idx2 = pipeline_tf2.add_block(tf_block_convert2uv, "RAW", G_NCHAN)
idx3 = pipeline_tf2.add_block(tf_block_convert2uv, idx2, G_NCHAN)
pipeline_tf2.make_block_output(idx2,'uV')
pipeline_tf2.make_block_output(idx3,'uV')
x2,y2 = pipeline_tf2.build_main_graph()
i1,o1 =  {x2:np.asarray([[i*x for i in range(10)] for x in range(G_NCHAN)])},y2

'''

FIR FILT TEST

'''

fir_coeffs = '7-14_BP_FIR_BLACKMANN.npy'
#b,a = np.load('C:\\Conda\\MartianBCI\\Demos\\RL_Utils\\Filters\\' + fir_coeffs)
b,a = np.load('C:\\Users\\marzipan\\workspace\\MartianBCI\\Demos\\RL_Utils\\Filters\\' + fir_coeffs)

Sin10 = np.asarray([[np.sin(2*np.pi*(x/250)*10) for x in range(1000)]]*G_NCHAN)
Sin25 = np.asarray([[np.sin(2*np.pi*(x/250)*25) for x in range(1000)]]*G_NCHAN)

pipeline_tf3 = Pipeline_TF(list(Sin25.shape))
fir_idx = pipeline_tf3.add_block(tf_block_fir, "RAW", G_NCHAN, b)
pipeline_tf3.make_block_output(fir_idx, "fir_flt")
x3,y3 = pipeline_tf3.build_main_graph()
i2,o2 = {x3:Sin25},y3


'''

BAND POWER TEST

'''

Sin0_15 = np.asarray([[np.sin(2*np.pi*(x/250)*i) for x in range(1000)] for i in range(G_NCHAN)])

pipeline_tf4 = Pipeline_TF(list(Sin0_15.shape))
bp_idx = pipeline_tf4.add_block(tf_block_band_powers, "RAW", G_NCHAN, 1000)
pipeline_tf4.make_block_output(bp_idx, "band_powers")
x4,y4 = pipeline_tf4.build_main_graph()
i3,o3 = {x4:Sin0_15},y4

'''

IAF TEST

'''
Sin0_15 = np.asarray([[np.sin(2*np.pi*(x/250)*i) for x in range(1000)] for i in range(G_NCHAN)])
pipeline_tf5 = Pipeline_TF(list(Sin0_15.shape))
iaf_idx = pipeline_tf5.add_block(tf_block_IAF, "RAW", G_NCHAN)
pipeline_tf5.make_block_output(iaf_idx, "iaf")
x5,y5 = pipeline_tf5.build_main_graph()
i4,o4 = {x5:Sin0_15},y5


'''

tf Session

'''
sess = tf.Session()
mWriter = tf.summary.FileWriter(".\\Logs\\",sess.graph)


p = sess.run(o0,i0)
q = sess.run(o1,i1)
r = sess.run(o2,i2)
s = sess.run(o3,i3)
t = sess.run(o4,i4)