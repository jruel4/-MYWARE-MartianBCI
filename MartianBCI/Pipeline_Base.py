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
from collections import deque
from pylsl import  StreamInlet, resolve_stream


if __name__ == "__main__":
    from Blocks.Block import Block
    from Blocks.Block_LSL import Block_LSL
else:
    from .Blocks.Block import Block
    from .Blocks.Block_LSL import Block_LSL

class NoInputBlock(Exception): pass

class PipelineBase:
    '''
    High level pipeline class for Real-time EEG DSP
    '''
    
    def __init__(self, _UIDStart=128000, _UIDEnd=256000):
        self.mUIDStart = _UIDStart
        self.mUIDEnd = _UIDEnd
        self.mBlocks = dict() # Signal processing blocks are primary data transformation operations
    
    def _block_uid_exists(self,block_uid):
        if block_uid not in self.mBlocks.keys():
             raise KeyError("block_uid does not exist in block list, " + block_uid)
        return True
    
    def _parent_uid_exists(self,parent_uid):
        if parent_uid != "RAW" and parent_uid not in self._blocks.keys():
             raise KeyError("parent_uid does not exist in block list or as \"RAW\": " + parent_uid)
    
    def _is_block_subclass(self,block):
        if not issubclass(block, Block):
            raise TypeError("Input object does not inherit from Block class")
    
    def _generate_block_uid(self):
        block_uid = np.random.randint(self.mUIDStart,self.mUIDEnd)
        while block_uid in self.mBlocks.keys(): block_uid = np.random.randint(self.mUIDStart,self.mUIDEnd)            
        return block_uid

    def add_block(self, bFunction, parentUID, *args, **kwargs):
        '''
        Add signal processing block with required initialization args.
        Ensuring type correctness of init args should be handled in bFunction.

        returns block UID
        
        Example usage: pipeline.add_block(test_block, ['1','2'],{'kw1':'3','kw2':'4'})
        '''

        self._parent_uid_exists(parentUID)
        self._is_block_subclass(bFunction)        

        block_function = bFunction(self, *args, **kwargs)            
        block_uid = self._generate_block_uid() 
        
        #Generate structure, prepend to blocklist
        new_block = {
                block_uid:{
                        'func':block_function,
                        'parent':parentUID,
                        }
                }
        self.mBlocks.update(new_block)
        return block_uid
    
    def execute_block(self, block_uid, _buf):
        '''
        This method recurses through available blocks
        '''
        buf = self.mBlocks[block_uid]['func'].run(_buf)
        for next_block_uid, next_block_attrib in self.mBlocks.items():
            if next_block_attrib['parent'] == block_uid:
                self.execute_block(next_block_uid, buf)
        return
            
    def execution_thread(self):
        '''
        Executes signal processing pipeline
        '''
        
        starting_block_uids = list()
        for block_uid,v in self.mBlocks.items():
            if v['parent'] == "RAW":
                starting_block_uids.append(block_uid)

        #Verify that we have at least one starting block
        if not starting_block_uids:
            raise NoInputBlock("No input block selected")
        
        new_count = 0
        buffer = deque(maxlen=self.inbuf_len)
        while self.run_thread_event.is_set():
            try:
                new_data = np.asarray(self._in_queue.get(block=True, timeout=0.004))
                buffer.append(new_data[self.chan_sel])
                new_count += 1
            except queue.Empty as e:
                pass
            if new_count >= self.sample_update_interval and len(buffer) >= self.inbuf_len: 
                new_count = 0
                
                '''
                This is where actual thread execution occurs
                '''
                for start_uid in starting_block_uids:
                    self.execute_block(start_uid, buffer)

    def run(self):
        '''
        Launch thread which executes signal processing pipeline until termination
        flag is set.
        '''
        if self.run_thread_event.is_set():
            print("Run thread is already active")
        else:
            self.run_thread_event.set()
            # Create threads
            acquisition_thread = Thread(target=self.acquisition_thread)
            run_thread = Thread(target=self.execution_thread)
            # Start threads
            acquisition_thread.start()
            run_thread.start()
        
    def stop(self):
        if not self.run_thread_event.is_set():
            print("There is no active run thread to stop")
        else: 
            self.run_thread_event.clear()