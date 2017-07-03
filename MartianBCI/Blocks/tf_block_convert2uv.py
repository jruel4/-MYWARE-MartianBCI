# -*- coding: utf-8 -*-

if __name__ == "__main__":
    from Block_TF import Block_TF
else:
    from .Block_TF import Block_TF

import tensorflow as tf


'''
tf_block_convert2uv

INIT:
    in:
        _NCHAN - Number of channels in incoming signal; used to validate input shape
        _GAIN - Gain used on amplifier (currently only supports single value)
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
            ['data'] - #electrodes x #samples, values scaled to uV
            ['summaries'] - unchanged
            ['updates'] - unchanged

GET_OUTPUT_DIM:
    in:
        _BUFLEN - Length of input (i.e. #samples), used to calculate the output
    ret:
        Length two array, [#chan, #samples]

'''


class tf_block_convert2uv (Block_TF):
    def __init__(self, _NCHAN, _GAIN = 24):
        self.mNCHAN = tf.constant(_NCHAN,shape=[])
        self.mGAIN = tf.constant(_GAIN,shape=[])
        self.mSCALE_FACTOR = tf.constant( ( ( (4.5/ (2**23 - 1)) / _GAIN) * 1e6),shape=[])

    def run(self, _buf):
        with tf.name_scope("RUN_Conv2uV"):
            tf.assert_rank(_buf['data'], 2, message="JCR: Input must be rank 2 tensor")
            asserts= [
                    tf.assert_equal(tf.shape(_buf['data'])[0], self.mNCHAN, message="JCR: Input Dim-0 must equal number of channels")
                    ]
    
            with tf.control_dependencies(asserts):
                dout=tf.multiply(_buf['data'], self.mSCALE_FACTOR)
                return {
                        'data':dout,
                        'summaries':_buf['summaries'],
                        'updates':_buf['updates']
                        }
    
    def get_output_dim(self, _BUFLEN):
        return self.mNCHAN * _BUFLEN