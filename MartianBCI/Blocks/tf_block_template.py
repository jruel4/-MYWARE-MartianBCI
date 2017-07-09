# -*- coding: utf-8 -*-

if __name__ == "__main__":
    from Block_TF import Block_TF
else:
    from .Block_TF import Block_TF

import tensorflow as tf


#NOTE: This is a template block. Need to better structure OOP of the pipeline, might use this.
#       For now, just copy/paste the template when creating a block, makes things easier

'''
tf_block_TEMPLATE

__init__():
    in:
        ...
    ret:
        none
run():
    in:
        _buf - dictionary containing:
            ['data']:
                first key: ...
            ['summaries']:
                list to be passed back to summary writer
            ['updates']:
                list of update/assign ops to evaluate at the end of the run
    ret:
        dictionary containing:
            ['data']:
                'key': ...
                ...
            ['summaries']:
                ...
            ['updates']:
                ...

get_output_struct():
    in:
        none
    ret:
        ['data']:
            ... 

example:
    __PIPE__.add_block(tf_block_TEMPLATE, ...)

'''


class tf_block_TEMPLATE (Block_TF):
    def __init__(self, _PIPE_TF):
        raise NotImplementedError("Error, template class should not actually be used!")
        self.mPipeTF = _PIPE_TF
        
        self.mInKeys = None

    def run(self, _buf):
        if self.mInKeys == None:
            self.mInKeys = super().get_input_keys(self.mPipeTF)
        with tf.name_scope("B_TEMPLATE"):
            input_data = _buf['data'][self.mInKeys[0]]

            dout=input_data
            
            return {
                    'data':{'TEMPLATE':dout},
                    'summaries':_buf['summaries'],
                    'updates':_buf['updates']
                    }

    def get_output_struct(self):
        return {
                'data':{'TEMPLATE':[-1]},
                'summaries':0,
                'updates':0
                }