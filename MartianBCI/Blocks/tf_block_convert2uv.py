# -*- coding: utf-8 -*-

if __name__ == "__main__":
    from Block_TF import Block_TF
else:
    from .Block_TF import Block_TF

import tensorflow as tf


'''
tf_block_convert2uv

__init__():
    in:
        _NCHAN - Number of channels in incoming signal; used to validate input shape
        _GAIN - Gain used on amplifier (currently only supports single value)
    ret:
        none
run():
    in:
        _buf - dictionary containing:
            ['data']:
                first key: #electrodes x #samples data buffer
            ['summaries'] - list to be passed back to summary writer
            ['updates'] - list of update/assign ops to evaluate at the end of the run
    ret:
        dictionary containing:
            ['data']:
                'uV' " #electrodes x #samples, values scaled to uV
            ['summaries'] - unchanged
            ['updates'] - unchanged

get_output_struct():
    in:
        none
    ret:
        ['data']:
            'uV' : [numberOfChannels, var_len], where var_len is -1 to show variable length 

'''


class tf_block_convert2uv (Block_TF):
    def __init__(self, _PIPE_TF, _NCHAN, _GAIN = 24):
        self.mPipeTF = _PIPE_TF
        self.mNCHAN_NON_TENSORFLOW = _NCHAN
        self.mNCHAN = tf.constant(_NCHAN,shape=[])
        self.mGAIN = tf.constant(_GAIN,shape=[])
        self.mSCALE_FACTOR = tf.constant( ( ( (4.5/ (2**23 - 1)) / _GAIN) * 1e6),shape=[])
        self.mInKeys = None

    def run(self, _buf):
        if self.mInKeys == None:
            self.mInKeys = super().get_input_keys(self.mPipeTF)
        with tf.name_scope("B_Convert2uV"):
            input_data = _buf['data'][self.mInKeys[0]]
            tf.assert_rank(input_data, 2, message="JCR: Input must be rank 2 tensor")
            asserts= [
                    tf.assert_equal(tf.shape(input_data)[0], self.mNCHAN,
                                    message="JCR: Input Dim-0 must equal number of channels")
                    ]
    
            with tf.control_dependencies(asserts):
                dout=tf.multiply(input_data[:], self.mSCALE_FACTOR)
                return {
                        'data':{'uV':dout},
                        'summaries':_buf['summaries'],
                        'updates':_buf['updates']
                        }
    
    def get_output_struct(self):
        return {
                'data':{'uV':[self.mNCHAN_NON_TENSORFLOW,-1]},
                'summaries':0,
                'updates':0
                }