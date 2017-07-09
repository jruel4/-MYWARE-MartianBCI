# -*- coding: utf-8 -*-
if __name__ == "__main__":
    from Block_TF import Block_TF
else:
    from .Block_TF import Block_TF

import tensorflow as tf
import numpy as np
from Demos.RL_Utils.Helpers import Processing_TF as Utils

''''
Usage;

example:
    _PIPELINE_TF_.add_block(tf_block_band_powers, _PARENT_UID="RAW",
                            _NCHAN=8, _SIGLEN=1000, _BIN_WIDTH=0.25)
'''

class tf_block_band_powers (Block_TF):
    def __init__(self, _PIPE_TF, _NCHAN, _SIGLEN, _FS=250.0, _BIN_WIDTH=0.5):
        self.mPipeTF = _PIPE_TF
        self.mNCHAN_NON_TENSORFLOW = _NCHAN
        self.mNCHAN = tf.constant(_NCHAN, shape=[],name='NCHAN')
        self.mFS = _FS
        self.mBIN_WIDTH = _BIN_WIDTH
        self.mSIGLEN = _SIGLEN
        self.mInKeys = None

    def run(self, _buf):
        if self.mInKeys == None:
            self.mInKeys = super().get_input_keys(self.mPipeTF)
        with tf.name_scope("B_BandPowers"):
            input_data = _buf['data'][self.mInKeys[0]]
            tf.assert_rank(input_data, 2, message="JCR: Input must be rank 2 tensor")
            asserts= [
                    tf.assert_equal(tf.shape(input_data)[0], self.mNCHAN,
                                    message="JCR: Input Dim-0 must equal number of channels")
                    ]

            with tf.control_dependencies(asserts):                
                hz_per_point = self.mFS / self.mSIGLEN
                points_per_bin = np.maximum( self.mBIN_WIDTH // hz_per_point, 1.0)
                
                total_number_bins = int(self.mSIGLEN // points_per_bin)
                total_number_points = int(total_number_bins * points_per_bin)
                f_start = 0
                f_end = total_number_points * self.mFS / self.mSIGLEN
                
                dout = Utils.extract_frequency_bins(input_data, f_start, f_end, total_number_bins, self.mFS, self.mSIGLEN)
                asserts = [
                            tf.assert_equal(tf.shape(dout)[0], self.mNCHAN,
                                            message="JCR: Input/output shape mismatch",name='FinalCheck')
                            ]
                with tf.control_dependencies(asserts):
                    return {
                            'data':{'band_powers':dout},
                            'summaries':_buf['summaries'],
                            #pass dout shape to updates so that the assert gets evaluated
                            'updates': _buf['updates'] + [tf.shape(dout)[0]]
                            }
    def get_output_struct(self):
        return {
                'data':{'band_powers':[self.mNCHAN_NON_TENSORFLOW,-1]},
                'summaries':0,
                'updates':[1]
                }


'''
#Useful for testing

Sin0_15 = np.asarray([[np.sin(2*np.pi*(x/250)*i) for x in range(1000)] for i in range(16)])
Sin10 = np.asarray([[np.sin(2*np.pi*(x/250)*10) for x in range(1000)]]*16)
Sin25 = np.asarray([[np.sin(2*np.pi*(x/250)*25) for x in range(1000)]]*16)
    

tf.reset_default_graph()
with tf.Session() as sess:
    Sin0_15_TF = tf.constant(Sin0_15, dtype=tf.float32)
    
    tb = tf_block_band_powers(16,1000)
    d = tb.run({'data':Sin0_15_TF,
                'summaries':[],
                'updates':[]})
    mWriter = tf.summary.FileWriter(".\\Logs\\",sess.graph)
    q= sess.run(d)
'''