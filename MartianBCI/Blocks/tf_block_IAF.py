# -*- coding: utf-8 -*-
if __name__ == "__main__":
    from Block_TF import Block_TF
else:
    from .Block_TF import Block_TF

import tensorflow as tf
import numpy as np
from Demos.RL_Utils.Helpers import Processing_TF as Utils

class tf_block_IAF (Block_TF):
    def __init__(self, _PIPE_TF, _NCHAN, _FS=250, _PEAK_WINDOW=0.5):
        self.mPipeTF = _PIPE_TF
        self.mNCHAN_NON_TENSORFLOW = _NCHAN
        self.mNCHAN = tf.constant(_NCHAN, shape=[],name='NCHAN')
        self.mFS = tf.constant(_FS,shape=[])
        self.mPEAK_WINDOW = tf.constant(_PEAK_WINDOW, shape=[])
        self.mInKeys = None

    def run(self, _buf):
        if self.mInKeys == None:
            self.mInKeys = super().get_input_keys(self.mPipeTF)
        with tf.name_scope("B_IAF"):
            input_data = _buf['data'][self.mInKeys[0]]
            tf.assert_rank(input_data, 2, message="JCR: Input must be rank 2 tensor")
            asserts= [
                    tf.assert_equal(tf.shape(input_data)[0], self.mNCHAN,
                                    message="Input Dim-0 must equal number of channels")
                    ]
            
            with tf.control_dependencies(asserts):
                s_len = tf.shape(input_data)[1]
                
                pphz = tf.realdiv(tf.cast(s_len, tf.float32) , tf.cast(self.mFS, tf.float32))
                
                #This value (in Hz) is used to determine the peak frequency - larger window uses more surrounding values to calculate IAF
                IAF_Peak_Window_Size = tf.realdiv(self.mPEAK_WINDOW, 2.0)
                
                asserts = [
                        tf.assert_greater_equal(IAF_Peak_Window_Size, tf.realdiv(1.0, pphz),
                                                message="Invalid number of Hz/Window")
                        ]
                with tf.control_dependencies(asserts):
                    IAF_Window = tf.ones([tf.cast(tf.multiply(IAF_Peak_Window_Size,pphz),tf.int32)])
            
                    data_fft = tf.fft(tf.cast(input_data,tf.complex64))
                    
                    #Convolve the window over the FFT to strengthen peak
                    data_fft_neighbor_avg = Utils.multi_ch_conv(tf.cast(tf.abs(data_fft),tf.float32),IAF_Window)
                    half_size = tf.cast((tf.shape(data_fft_neighbor_avg)[1] / 2), tf.int32)
                    dout1 = tf.argmax(data_fft_neighbor_avg[:, 0:half_size], axis=1)
                    dout = tf.realdiv( tf.cast(dout1, tf.float32), tf.realdiv(tf.cast(half_size,tf.float32), tf.realdiv(tf.cast(self.mFS,tf.float32), 2.0)))
        
                    print(tf.shape(dout))
                    print(self.mNCHAN)
                    asserts = [
                            tf.assert_equal(tf.shape(dout)[0], self.mNCHAN,
                                            message="Input/output shape mismatch",name='FinalCheck')
                            ]
                    with tf.control_dependencies(asserts):
                        tf.summary.histogram("IAF", dout)
                        return {
                                'data':{'iaf':dout},
                                'summaries':_buf['summaries'],
                                #pass dout shape to updates so that the assert gets evaluated
                                'updates': _buf['updates'] + [tf.shape(dout)[0]]
                                }
    
    def get_output_struct(self):
        return {
                'data':{'iaf':self.mNCHAN_NON_TENSORFLOW},
                'summaries':0,
                'updates':[1]
                }


'''
#Useful for testing

Sin10 = np.asarray([[np.sin(2*np.pi*(x/250)*i) for x in range(1000)] for i in range(16)])
Sin25 = np.asarray([[np.sin(2*np.pi*(x/250)*25) for x in range(1000)]]*16)
    
tf.reset_default_graph()
with tf.Session() as sess:
    tb = tf_block_IAF(16)
    d = tb.run({'data':Sin10.astype(np.float32),
                'summaries':[],
                'updates':[]})
    mWriter = tf.summary.FileWriter(".\\Logs\\",sess.graph)
    q= sess.run(d)
'''