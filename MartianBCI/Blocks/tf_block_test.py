# -*- coding: utf-8 -*-


from MartianBCI.Blocks.Block_TF import Block_TF
import numpy as np
import tensorflow as tf

class tf_block_test (Block_TF):
        
        def __init__(self, _pipe_tf, pos1, pos2, kw1 = "val1", kw2 = "val2"):
            self.mPipeTF = _pipe_tf
            print("pos1: ",pos1)
            print("pos2: ",pos2)
            print("kw1: ",kw1)
            print("kw2: ",kw2)
            self.idx = tf.constant(0.0)

            
        def run(self, buf):
            with tf.name_scope("B_TEST"):
                input_keys = super().get_input_keys(self.mPipeTF)
                self.idx = self.idx + 1
                self.idx2 = self.idx + buf['data'][input_keys[0]]
                return {
                    'data':{'x':[x*self.idx2 for x in range(8)]},
                    'summaries':buf['summaries'],
                    'updates':buf['updates']
                    }
        def get_output_struct(self):
            return {
                'data':{'x':-1},
                'summaries':0,
                'updates':0
                }