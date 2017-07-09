# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 00:19:38 2017

@author: marzipan
"""

#from MartianBCI.Pipeline_TF import Pipeline_TF

class Block_TF (object):
    '''
    Abstract class for Pipeline, indicates required methods
    '''
    def __init__(self,_pipe_tf):
        raise NotImplementedError( "Blocks must implement init() method with positional pipe argument." )

    def _get_parent_output_struct(self,_pipe_tf):
#        if not isinstance(_pipe_tf, Pipeline_TF):
#            raise TypeError("Expected input pipeline to be of type Pipeline_TF")
        if _pipe_tf != []:
            pipeline = _pipe_tf
        else:
            pipeline = self.mPipeTF
        self.mBlockUID = pipeline._get_block_uid(self)
        self.mParentUID = pipeline._get_parent_uid(self.mBlockUID)
        if self.mParentUID == "RAW":
            self.mInputKeys = ["RAW"]
        else:
            self.mInputKeys = list(pipeline.mBlocks[self.mParentUID]['func'].get_output_struct()['data'].keys())
    
    def run(self):
        raise NotImplementedError( "Blocks must implement run() method." )
    
    def get_output_struct(self):
        raise NotImplementedError( "Blocks must implement get_output_struct() method." )
            

    def get_input_keys(self, _pipe_tf):
        self._get_parent_output_struct(_pipe_tf)
        return self.mInputKeys
        
        
