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

Call this initially to create a processing graph
Call run to process a sample through the graph


'''

def tf_block_adapter_add(bFunction, *args, **kwargs):
        '''
        Add signal processing block with required initialization args.
        Ensuring type correctness of init args should be handled in bFunction.
        
        Example usage: pipeline.add_block(test_block, ['1','2'],{'kw1':'3','kw2':'4'})
        '''
        block_instance = bFunction(*args, **kwargs)
        assert isinstance(block_instance, Block_TF)
        return block_instance
    
    
class tf_block_adapter_io (Block):
    def __init__(self, _pipe, _blocks, _LEN, _NCHAN, _WRITE_GRAPH=True, _WRITE_SUMMARIES=False, _WRITE_METADATA_N_STEPS=0):
        self.mPipe = _pipe
        self.mBlocks = _blocks
        self.mWRITE_GRAPH = _WRITE_GRAPH
        self.mWRITE_SUMMARIES = _WRITE_SUMMARIES
        self.mMETADATA_FREQ = _WRITE_METADATA_N_STEPS
        
        #Create out input buffer
        self.mLEN = _LEN
        self.mNCHAN = _NCHAN
        self.mBuf = np.zeros([self.mLEN, self.mNCHAN])
        
        #Step counter, for logging metadata
        self.mStep = 1

        #Construct TF graph
        self.mGraphIn,self.mGraphOut = self.build_main_graph()

        #Init session, log graph to file
        self.mSess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        print("Initializing global variables...", self.mSess.run( tf.global_variables_initializer() ))
        if self.mWRITE_GRAPH:
            self.mWriter = tf.summary.FileWriter(".\\Logs\\",self.mSess.graph)
        return

    def build_main_graph(self):
        #TODO: Add shape checking on input
        indata = tf.placeholder(tf.float32,shape=[self.mNCHAN, self.mLEN])
        
        #Use buf initially as an input
        buf = {
                'data':indata,
                'summaries':[],
                'updates':[]
                }
        for block in self.mBlocks:
            buf = block.run(buf)
        
        #Then as our ouput
        outdata = buf
        
        return indata, outdata

    def run(self, _buf, test=False):
        #remove the oldest value from the buffer
        self.mBuf = np.delete(self.mBuf,0,0)
        #and insert the next
        self.mBuf = np.append(self.mBuf,[_buf.pop()],0)

        #add metadata to the graph
        if (self.mWRITE_SUMMARIES) and (self.mMETADATA_FREQ >= 1) and (np.mod(self.mMETADATA_FREQ,self.mStep) == 0):
            print('Adding run metadata for step: ', self.mStep)
            run_options = tf.RunOptions(trace_level=tf.RunOptions.SOFTWARE_TRACE)
            run_metadata = tf.RunMetadata()
            output = self.mSess.run(
                    self.mGraphOut,
                    feed_dict={self.mGraphIn: np.transpose(self.mBuf)},
                    options=run_options,
                    run_metadata=run_metadata)
            self.mWriter.add_run_metadata(run_metadata, 'S: %d' % self.mStep)
            self.mWriter.add_summary(output['summaries'], global_step=self.mStep)
        else:
            output = self.mSess.run(self.mGraphOut, feed_dict={self.mGraphIn: np.transpose(self.mBuf)})
            if (self.mWRITE_SUMMARIES):
                self.mWriter.add_summary(output['summaries'], global_step=self.mStep)
                
        self.mStep += 1
        return np.transpose(output['data'])[np.mod(self.mStep,250)]
    
    def get_output_dim(self, buf_len, chan_sel):
        return 8 #TODO make this dynamic