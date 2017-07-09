# -*- coding: utf-8 -*-

if __name__ == "__main__":
    from Block_TF import Block_TF
else:
    from .Block_TF import Block_TF

import tensorflow as tf
import numpy as np
from Demos.RL_Utils.Helpers import Processing_TF as Utils


'''
tf_block_fir

INIT:
    in:
        _NCHAN - Number of channels in incoming signal; used to validate input shape
        _COEFFS - List of coefficients of filter, 1D
    ret:
        None
RUN:
    in:
        _buf - dictionary containing:
            ['data'] - #electrodes x #samples data buffer
            ['summaries'] - list to be passed back to summary writer
            ['updates'] - list of update/assign ops to evaluate at the end of the run
    ret:
        dictionary containing:
            ['data'] - #electrodes x #samples, filtered
            ['summaries'] - unchanged
            ['updates'] - unchanged

GET_OUTPUT_DIM:
    in:
        _BUFLEN - Length of input (i.e. #samples), used to calculate the output
    ret:
        Length two array, [#chan, #samples]

'''

class tf_block_fir (Block_TF):
    def __init__(self, _PIPE_TF, _NCHAN, _COEFFS):
        self.mPipeTF = _PIPE_TF
        self.mNCHAN_NON_TENSORFLOW = _NCHAN
        self.mNCHAN = tf.constant(_NCHAN,shape=[])
        self.mCOEFFS = tf.constant(_COEFFS)
        self.mInKeys = None

        tf.assert_rank(self.mCOEFFS, 1,
                       message="JCR: Coeffs must be rank 1 tensor",
                       name='FIRCoeffInputCheck')

    def run(self, _buf):
        if self.mInKeys == None:
            self.mInKeys = super().get_input_keys(self.mPipeTF)
        with tf.name_scope("B_FIRFilt"):
            input_data = _buf['data'][self.mInKeys[0]]
            tf.assert_rank(input_data, 2, message="JCR: Input must be rank 2 tensor")
            asserts= [
                    tf.assert_equal(tf.shape(input_data)[0], self.mNCHAN,
                                    message="JCR: Input Dim-0 must equal number of channels")
                    ]
    
            with tf.control_dependencies(asserts):
                dout = Utils.multi_ch_conv(input_data, self.mCOEFFS)
                return {
                        'data':{'fir_flt':dout},
                        'summaries':_buf['summaries'],
                        'updates':_buf['updates']
                        }

    def get_output_struct(self):
        return {
                'data':{'fir_flt':[self.mNCHAN_NON_TENSORFLOW,-1]},
                'summaries':0,
                'updates':0
                }

'''
#Useful for testing

fir_coeffs = '7-14_BP_FIR_BLACKMANN.npy'
#b,a = np.load('C:\\Conda\\MartianBCI\\Demos\\RL_Utils\\Filters\\' + fir_coeffs)
b,a = np.load('C:\\Users\\marzipan\\workspace\\MartianBCI\\Demos\\RL_Utils\\Filters\\' + fir_coeffs)

Sin10 = np.asarray([[np.sin(2*np.pi*(x/250)*10) for x in range(1000)]]*16)
Sin25 = np.asarray([[np.sin(2*np.pi*(x/250)*25) for x in range(1000)]]*16)
    
tf.reset_default_graph()
with tf.Session() as sess:
    tb = tf_block_fir(16,(np.asarray(b)).astype(np.float32))
    d = tb.run({'data':Sin10.astype(np.float32),
                'summaries':[],
                'updates':[]})
    q= sess.run(d)
'''