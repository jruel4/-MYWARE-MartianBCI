# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 00:19:38 2017

@author: marzipan
"""

class Block (object):
    '''
    Abstract class for Pipeline, indicates required methods
    '''
    def __init__(self, _pipe):
        raise NotImplementedError( "Blocks must implement init() method with positional pipe argument." )
    
    
    def _get_parent_output_struct(self,_pipe):
        if _pipe != []:
            pipeline = _pipe
        else:
            pipeline = self.mPipe
        self.mBlockUID = pipeline._get_block_uid(self)
        self.mParentUID = pipeline._get_parent_uid(self.mBlockUID)
        if self.mParentUID == "RAW":
            self.mInputKeys = ["RAW"]
        else:
            self.mInputKeys = list(pipeline.mBlocks[self.mParentUID]['func'].get_output_struct().keys())
    
    
    def run(self,_buf):
        raise NotImplementedError( "Blocks must implement run() method." )
    
    def get_default(self, _buf):
        return _buf['default']
    
    def get_output_struct(self):
        raise NotImplementedError( "Blocks must implement get_output_dim() method." )
        
    def get_input_keys(self, _pipe):
        self._get_parent_output_struct(_pipe)
        return self.mInputKeys