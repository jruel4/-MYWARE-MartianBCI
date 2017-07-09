# -*- coding: utf-8 -*-

if __name__ == "__main__":
    from Block_TF import Block_TF
else:
    from .Block_TF import Block_TF

import tensorflow as tf


'''
tf_block_slice

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


class tf_block_slice (Block_TF):
    def __init__(self, _PIPE_TF, _BEGIN, _SIZE, _OUTPUT_LEN):
        self.mPipeTF = _PIPE_TF
        self.mBEGIN = _BEGIN
        self.mSIZE = _SIZE
        self.mOUTPUT_LEN = _OUTPUT_LEN
        
        if not isinstance(self.mBEGIN, list) or not isinstance(self.mSIZE, list):
            raise TypeError("Beginning and size must be arrays corresponding to start position and size of output slice")

        self.mInKeys = None

    def run(self, _buf):
        if self.mInKeys == None:
            self.mInKeys = super().get_input_keys(self.mPipeTF)
        with tf.name_scope("B_SliceOut"):
            input_data = _buf['data'][self.mInKeys[0]]
            tf.assert_rank_at_least(input_data, 1,
                                    message="ReduceDims: Input must be rank at least rank 1 tensor")
            dout=tf.slice(input_data, self.mBEGIN, self.mSIZE)
            
            return {
                    'data':{'slice':dout},
                    'summaries':_buf['summaries'],
                    'updates':_buf['updates']
                    }

    def get_output_struct(self):
        return {
                'data':{'slice':self.mOUTPUT_LEN},
                'summaries':0,
                'updates':0
                }