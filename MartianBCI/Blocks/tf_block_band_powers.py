# -*- coding: utf-8 -*-
if __name__ == "__main__":
    from Block_TF import Block_TF
else:
    from .Block_TF import Block_TF

import tensorflow as tf
import numpy as np
from Demos.RL_Utils.Helpers import Processing_TF as Utils

class tf_block_band_powers (Block_TF):
    def __init__(self, _NCHAN, _SIGLEN, _FS=250.0, _BIN_WIDTH=0.75):
        self.mNCHAN = tf.constant(_NCHAN, shape=[],name='NCHAN')
        self.mFS = _FS
        self.mBIN_WIDTH = _BIN_WIDTH
        self.mSIGLEN = _SIGLEN

    def run(self, _buf):
        with tf.name_scope("RUN_BandPower"):
            tf.assert_rank(_buf['data'], 2, message="JCR: Input must be rank 2 tensor")
            asserts= [
                    tf.assert_equal(tf.shape(_buf['data'])[0], self.mNCHAN, message="JCR: Input Dim-0 must equal number of channels")
                    ]

            with tf.control_dependencies(asserts):                
                hz_per_point = self.mFS / self.mSIGLEN
                points_per_bin = np.maximum( self.mBIN_WIDTH // hz_per_point, 1.0)
                
                total_number_bins = int(self.mSIGLEN // points_per_bin)
                total_number_points = int(total_number_bins * points_per_bin)
                f_start = 0
                f_end = total_number_points * self.mFS / self.mSIGLEN
                
                dout = Utils.extract_frequency_bins(_buf['data'], f_start, f_end, total_number_bins, self.mSIGLEN, self.mFS)
                asserts = [
                            tf.assert_equal(tf.shape(dout)[0], self.mNCHAN, message="JCR: Input/output shape mismatch",name='FinalCheck')
                            ]
                with tf.control_dependencies(asserts):
                    return {
                            'data':dout,
                            'summaries':_buf['summaries'],
                            #pass dout shape to updates so that the assert gets evaluated
                            'updates': _buf['updates'] + [tf.shape(dout)[0]]
                            }
    
    def get_output_dim(self):
        return self.mNCHAN






'''
#Useful for testing

Sin0_15 = np.asarray([[np.sin(2*np.pi*(x/250)*i) for x in range(1000)] for i in range(16)])
Sin10 = np.asarray([[np.sin(2*np.pi*(x/250)*10) for x in range(1000)]]*16)
Sin25 = np.asarray([[np.sin(2*np.pi*(x/250)*25) for x in range(1000)]]*16)
    

tf.reset_default_graph()
with tf.Session() as sess:
    Sin0_15_TF = tf.constant(Sin0_15, dtype=tf.float32)
    
    tb = tf_block_band_powers(16)
    d = tb.run({'data':Sin0_15_TF,
                'summaries':[],
                'updates':[]})
    mWriter = tf.summary.FileWriter(".\\Logs\\",sess.graph)
    q= sess.run(d)
'''