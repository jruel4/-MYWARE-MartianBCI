# -*- coding: utf-8 -*-


from Demos.RL_Utils.Helpers import Processing_TF as Utils


if __name__ == "__main__":
    from Block_TF import Block_TF
else:
    from .Block_TF import Block_TF

import tensorflow as tf
from scipy import signal



'''
tf_block_specgram

Returns spectrogram of input data using a tukey window (alpha set to 0.5)

__init__():
    in:
        _NCHAN              - Number of input channels
        _INPUT_LEN          - Length of input signal
        _WINDOW_STRIDE      - Amount to shift window each step
        _WINDOW_LEN = 500   - Length of window (optional if _WINDOW is provided)
        _WINDOW = []        - Optionally provide your own windowing function
        _ZERO_MEAN=True     - Optionally zero-mean the data (UNTESTED!)
    ret:
        none
run():
    in:
        _buf - dictionary containing:
            ['data']:
                first key: #electrodes x #samples data buffer
            ['summaries']:
                list to be passed back to summary writer
            ['updates']:
                list of update/assign ops to evaluate at the end of the run
    ret:
        dictionary containing:
            ['data']:
                'specgram': #windows x #channels x #frequencies output tensor (full tensor, even mirror)
            ['summaries']:
                none
            ['updates']:
                none

get_output_struct():
    in:
        none
    ret:
        ['data']:
            'specgram': [#windows, #channels, #frequencies]

example:
    __PIPE__.add_block(tf_block_specgram, _PARENT_UID="RAW",
                       _NCHAN=8, _INPUT_LEN=1000,
                       _WINDOW_LEN=500, _WINDOW_STRIDE=50)
    
example (custom window):
    from scipy import signal
    signal.hamming(100)
    __PIPE__.add_block(tf_block_specgram, _PARENT_UID="RAW",
                       _NCHAN=8, _INPUT_LEN=1000,
                       _WINDOW_LEN=500, _WINDOW_STRIDE=50)

'''

class AmbiguousParameter(Exception): pass

class tf_block_specgram (Block_TF):
    def __init__(self, _PIPE_TF, _NCHAN, _INPUT_LEN, _WINDOW_STRIDE, _WINDOW_LEN=0, _WINDOW=[], _ZERO_MEAN=False):
        self.mPipeTF = _PIPE_TF
        self.mNCHAN= _NCHAN
        self.mSIGLEN = _INPUT_LEN
        self.mWINDOW_STRIDE = _WINDOW_STRIDE
        self.mZERO_MEAN = _ZERO_MEAN
        
        #Use custom window
        if (_WINDOW_LEN == 0):
            if(_WINDOW == []):
                raise AmbiguousParameter("No window length or custom window provided")
            elif (len(_WINDOW) > _INPUT_LEN):
                raise ValueError("Window length: ",len(_WINDOW)," must be smaller than input length: ", _INPUT_LEN)
            self.mWINDOW = _WINDOW
            self.mWINDOW_LEN = len(_WINDOW)

        #Use Tukey window
        elif (_WINDOW == []):
            if (_WINDOW_LEN > _INPUT_LEN):
                raise ValueError("Window length: ",_WINDOW_LEN," must be smaller than input length: ", _INPUT_LEN)
            self.mWINDOW = signal.tukey(_WINDOW_LEN)
            self.mWINDOW_LEN = _WINDOW_LEN

        #Too many parameters provided
        else:
            raise AmbiguousParameter("Both window length and custom window parameters provided, only one should be provided")
        
        #Spectrogram length is the number of times we shift the window + 1 (for initial FFT)
        self.mSpecgramLen = ((self.mSIGLEN - self.mWINDOW_LEN) // self.mWINDOW_STRIDE) + 1
        print("Spectrogram Length: ", self.mSpecgramLen)
        self.mInKeys = None

    def run(self, _buf):
        if self.mInKeys == None:
            self.mInKeys = super().get_input_keys(self.mPipeTF)
        with tf.name_scope("B_Specgram"):
            input_data = _buf['data'][self.mInKeys[0]]
            tf.assert_rank(input_data, 2, message="JCR: Input must be rank 2 tensor")
            asserts= [
                    tf.assert_equal(tf.shape(input_data)[0], self.mNCHAN,
                                    message="JCR: Input Dim-0 must equal number of channels")
                    ]
    
            with tf.control_dependencies(asserts):
    
                #Pad window to same length of input signal
#                window_tf = tf.pad( tf.constant(self.mWINDOW,dtype=tf.float32), [[0,(self.mSIGLEN - self.mWINDOW_LEN)]], mode='CONSTANT')
                window_tf = tf.constant(self.mWINDOW,dtype=tf.float32, shape=[self.mWINDOW_LEN])

                #Tile window, one tile for each input channel
                window_tf = tf.expand_dims(window_tf,0)
                w_var = tf.tile(window_tf,[self.mNCHAN,1])
            
            
                #Zero-mean the data (optional)
                if self.mZERO_MEAN:
                    raise NotImplementedError("Pretty sure this works but untested. Feel free to comment this line out and try it out.")
                    mean = tf.reduce_mean(input_data,axis=1,keep_dims=True)
                    s_var = tf.subtract(input_data, mean)
                else:
                    s_var = input_data
                
                #Initialize our loop variables
                loop_idx = tf.constant(0,tf.int32)    
                loop_specgram_init = tf.zeros(shape=[self.mSpecgramLen, self.mNCHAN, self.mWINDOW_LEN], dtype=tf.complex64 )
                loopvars = [loop_idx,loop_specgram_init]
                
                #Define while loop to shift window, apply to data, take fft, and add to spectrogram
                def bod( _IDX, _SPECGRAM):
                    input_shifted = tf.slice(s_var, [0,tf.cast(self.mWINDOW_STRIDE * _IDX,tf.int32)], [-1,self.mWINDOW_LEN])
#                    window_shifted = Utils.shift_2d(w_var, tf.cast(self.mWINDOW_STRIDE * _IDX,tf.int32),1)
                    input_windowed = tf.multiply(input_shifted, w_var)
                    fft_tmp = tf.fft(tf.cast(input_windowed,tf.complex64))
                    fft_padded = tf.pad(tf.expand_dims(fft_tmp,0),[[_IDX, self.mSpecgramLen - _IDX - 1],[0,0],[0,0]])
                    
                    #Update loop variables
                    with tf.control_dependencies([fft_padded]):
                        _IDX = _IDX + 1
                        _SPECGRAM=tf.add(_SPECGRAM, fft_padded)
                    
                        return [_IDX,_SPECGRAM]
                
                #total_len - window_len  > (stride*(idx))
                def cond(_IDX, _SPECGRAM):
                    return tf.greater( self.mSIGLEN - self.mWINDOW_LEN, tf.cast(self.mWINDOW_STRIDE * _IDX, tf.int32 ))
                    
                dout = tf.while_loop(cond,bod,loopvars,parallel_iterations=100000)
                
                return {
                        'data':{'specgram':dout[1]},
                        'summaries':_buf['summaries'],
                        'updates':_buf['updates']
                        }

    def get_output_struct(self):
        return {
                'data':{'specgram':[self.mSpecgramLen, self.mNCHAN, self.mSIGLEN]},
                'summaries':0,
                'updates':0
                }