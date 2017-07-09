# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 22:17:14 2017

@author: marzipan
"""

# Define imports
import pylsl
from threading import Thread, Event
from queue import Queue
import queue
import time
import numpy as np
import tensorflow as tf
from collections import deque
from pylsl import  StreamInlet, resolve_stream


if __name__ == "__main__":
    from Blocks.Block import Block
    from Blocks.Block_TF import Block_TF
    from Blocks.Block_LSL import Block_LSL
else:
    from .Blocks.Block import Block
    from .Blocks.Block_TF import Block_TF
    from .Blocks.Block_LSL import Block_LSL

class NoInputBlock(Exception): pass


class Pipeline_TF():
    '''
    High level pipeline class for Real-time EEG DSP
    '''
    
    def __init__(self, _INPUT_SHAPE):

        '''
        mBlocks is dictionary list of blocks & parents
        mOutputs is dictionary list of blocks, block output key, block output lenght, and pipeline output key

        mIndata is the input of the main graph
        mOutdata is the output of the main graph
        
        mInputShape is used to create placeholder (allows for checking shape of expected input v. actual input)
        '''

        self.mBlocks = dict()
        self.mOutputs = dict()
        
        self.mOutdata = {
                'data':dict(),
                'summaries':dict(),
                'updates':dict()
                }
        self.mIndata = {
                'data':dict(),
                'summaries':dict(),
                'updates':dict()
                }
        
        #Create input buffer
        if not isinstance(_INPUT_SHAPE, list):
            raise TypeError("Input shape must be of type list")
        self.mInputShape = _INPUT_SHAPE
        
         
    def add_block(self, _BLOCK, _PARENT_UID, *args, **kwargs):
        '''
        Add signal processing block with required initialization args.
        Ensuring type correctness of init args should be handled in _BLOCK.

        returns block UID
        
        Example usage: pipeline.add_block(test_block, ['1','2'],{'kw1':'3','kw2':'4'})
        '''



        #Verify that _PARENT_UID is either present or we're using raw data
        if _PARENT_UID != "RAW" and _PARENT_UID not in self.mBlocks.keys():
             raise KeyError("TF: _PARENT_UID does not exist in block list" + _PARENT_UID)
        
        #Verify that _BLOCK is valid
        if issubclass(_BLOCK, Block_TF):
            block_function = _BLOCK(self, *args, **kwargs)
        else:
            print(_BLOCK)
            raise TypeError("TF: Input _BLOCK object does not inherit from Block_TF class, BLOCK:", _BLOCK)
            
        #Generate UID, verify that it doesn't already exist
        block_uid = np.random.randint(128000,256000)
        while block_uid in self.mBlocks.keys(): block_uid = np.random.randint(128000,256000)
        
        #Generate structure, prepend to blocklist
        new_block = {
                block_uid:{
                        'func':block_function,
                        'parent':_PARENT_UID,
                        }
                }
        self.mBlocks.update(new_block)

        return block_uid
    
    def make_block_output(self, _BLOCK_UID, _BLOCK_OUTPUT_KEY):
        '''
        Add TF block output to output of whole TF pipeline
        
        return the pipeline output key, used to retrieve specific data
        NOTE: pipeline output key is same as _BLOCK_OUTPUT_KEY IFF _BLOCK_OUTPUT_KEY
        does not already exist; if it does, then pipeline output key = _BLOCK_OUTPUT_KEY + "_1"
        '''
        
        block = self.mBlocks[_BLOCK_UID]
        block_output_keys = block['func'].get_output_struct()['data'].keys()

        if _BLOCK_OUTPUT_KEY not in block_output_keys:
            raise KeyError("TF: _BLOCK_OUTPUT_KEY: " + str(_BLOCK_OUTPUT_KEY) + " is invalid, valid keys are: " + str(block_output_keys))
        
        block_output_len = block['func'].get_output_struct()['data'][_BLOCK_OUTPUT_KEY]
        pipeline_output_key = _BLOCK_OUTPUT_KEY
        while pipeline_output_key in [x['pipeline_output_key'] for x in self.mOutputs.values()]:
            pipeline_output_key = pipeline_output_key + "_1"
        
        
        new_output={
                _BLOCK_UID:{
                        'block_output_key':_BLOCK_OUTPUT_KEY,
                        'block_output_len':block_output_len,
                        'pipeline_output_key':pipeline_output_key
                        }
                }
        self.mOutputs.update(new_output)
        return

    def _execute_block(self, _BLOCK_UID, _BUF):
        buf = self.mBlocks[_BLOCK_UID]['func'].run(_BUF)
        
        #Used for summaries and for variables updated every run
        self.mOutdata['summaries'].update({_BLOCK_UID:buf['summaries']})
        self.mOutdata['updates'].update({_BLOCK_UID:buf['updates']})

        #OUTPUTS
        if _BLOCK_UID in self.mOutputs:
            _BLOCK_OUTPUT_KEY = self.mOutputs[_BLOCK_UID]['block_output_key']
            out_tmp = { self.mOutputs[_BLOCK_UID]['pipeline_output_key']:buf['data'][_BLOCK_OUTPUT_KEY] }
            self.mOutdata['data'].update(out_tmp)
        
        #Recurse down into sub-branches
        for next__BLOCK_UID, next_block_attrib in self.mBlocks.items():
            if next_block_attrib['parent'] == _BLOCK_UID:
                self._execute_block(next__BLOCK_UID, buf)

    '''
    This function can be called internally from a block; pass the argument self
    to get back the UID, which can be used to get parent UID
    '''
    def _get_block_uid(self, _BLOCK):
        for x in self.mBlocks.keys():
            if self.mBlocks[x]['func'] == _BLOCK:
                return x
        
    def _get_parent_uid(self, _BLOCK_UID):
        return self.mBlocks[_BLOCK_UID]['parent']

    def build_main_graph(self):
        #TODO: Add shape checking on input
        self.mIndata = tf.placeholder(tf.float32,shape=self.mInputShape)
        
        #Use buf initially as an input
        buf = {
                'data':{"RAW":self.mIndata},
                'summaries':[],
                'updates':[]
                }
        #
        self.starting_block_uids = list()
        for block_uid,v in self.mBlocks.items():
            if v['parent'] == "RAW":
                self.starting_block_uids.append(block_uid)

        #Verify that we have at least one starting block
        if not self.starting_block_uids:
            raise NoInputBlock("No input block selected")
        
        #Build actual graph
        for start_uid in self.starting_block_uids:
            self._execute_block(start_uid, buf)
        
        '''
        Generate summaries
        So, summaries don't actually have to be passed around using the buffer dict
        Summaries should can just be generated here
        Forgot this when architecting, so ['summaries'] passed between nodes are completely useless
        Will fix, sometime...
        '''
        summaries = tf.summary.merge_all()
        if summaries != None:
            self.mOutdata['summaries'] = summaries

        return self.mIndata, self.mOutdata
    
    def get_input_struct(self):
        return {'rank':[2]}
    
    def get_output_struct(self):
        output_struct = dict()
        for x in self.mOutputs.values():
            output_struct_tmp  = {x['pipeline_output_key']:x['block_output_len']}
            output_struct.update(output_struct_tmp)
        return output_struct