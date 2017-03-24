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
    
    def run(self):
        raise NotImplementedError( "Blocks must implement run() method." )
    
    def get_output_dim(self):
        raise NotImplementedError( "Blocks must implement get_output_dim() method." )