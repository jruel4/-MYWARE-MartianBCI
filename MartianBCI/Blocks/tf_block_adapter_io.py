# -*- coding: utf-8 -*-
if __name__ == "__main__":
    from Block import Block
    from Block_TF import Block_TF
else:
    from .Block import Block
    from .Block_TF import Block_TF
    
import tensorflow as tf
import numpy as np


'''
tf_block_adapter_io



'''

class NoInputBlock(Exception): pass

class tf_block_adapter_io (Block):
    def __init__(self, _pipe, _PIPE_TF, _INPUT_SHAPE, _WRITE_GRAPH=True, _WRITE_SUMMARIES=False, _WRITE_METADATA_N_STEPS=0):
        self.mPipe = _pipe
        self.mPipeTF = _PIPE_TF
        self.mWRITE_GRAPH = _WRITE_GRAPH
        self.mWRITE_SUMMARIES = _WRITE_SUMMARIES
        self.mMETADATA_FREQ = _WRITE_METADATA_N_STEPS
        
        #Create out input buffer
        if not isinstance(_INPUT_SHAPE, list):
            raise TypeError("Input shape must be of type list")
        self.mInputShape = _INPUT_SHAPE
        self.mBuf = np.zeros(self.mInputShape)
        
        #Step counter, for logging metadata
        self.mStep = 1

        #Construct TF graph
        self.mGraphIn,self.mGraphOut = self.mPipeTF.build_main_graph()

        #Init session, log graph to file
#        config = tf.ConfigProto()
#        config.gpu_options.allow_growth = True
#        self.mSess = tf.Session(config=config)

        self.mSess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        print("Initializing global variables...", self.mSess.run( tf.global_variables_initializer() ))
        if self.mWRITE_GRAPH:
            self.mWriter = tf.summary.FileWriter(".\\Logs\\",self.mSess.graph)
        return

    def run(self, _buf, test=False):
        #remove the oldest value from the buffer
        self.mBuf = np.delete(self.mBuf,0,1)
        #and insert the next (& format input to be #chan x #samples)
        _buf_in = np.asarray([[x] for x in _buf.pop()])
        self.mBuf = np.append(self.mBuf,_buf_in,1)

        #add metadata to the graph
        if (self.mWRITE_SUMMARIES) and (self.mMETADATA_FREQ >= 1) and (np.mod(self.mStep,self.mMETADATA_FREQ) == 0):
            print('Adding run metadata for step: ', self.mStep)
            run_options = tf.RunOptions(trace_level=tf.RunOptions.SOFTWARE_TRACE)
            run_metadata = tf.RunMetadata()
            output = self.mSess.run(
                    self.mGraphOut,
                    feed_dict={self.mGraphIn: self.mBuf},
                    options=run_options,
                    run_metadata=run_metadata)
            self.mWriter.add_run_metadata(run_metadata, 'S: %d' % self.mStep)
            self.mWriter.add_summary(output['summaries'], global_step=self.mStep)
        else:
            output = self.mSess.run(self.mGraphOut, feed_dict={self.mGraphIn: self.mBuf})
            if (self.mWRITE_SUMMARIES):
                self.mWriter.add_summary(output['summaries'], global_step=self.mStep)
                
        self.mStep += 1
        return output['data']
    
    def get_output_struct(self):
        return self.mPipeTF.get_output_struct()
    