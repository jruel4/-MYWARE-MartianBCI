# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 22:17:14 2017

@author: marzipan
"""

# Define imports
import pylsl
from threading import Thread, Event
from Queue import Queue, Empty
#from multiprocessing import Queue
#import queue
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

'''
Pipeline - High level pipeline class for real-time DSP.

PIPELINE USAGE

Pipeline consists of "blocks" (functions) which each accept a single
input and have a single, labeled output. End blocks are LSL outlet streams.

Input blocks are specified with "RAW" for _PARENT_UID

Output blocks are special blocks which are all instances of Block_LSL (which inherits from meta-class Block).

Blocks are added to the pipeline using add_block

Pipeline relies exclusively on LSL for I/O. Pipeline class handles input, Block_LSL blocks handle output



Pipeline:
    add_block():
        inputs:
            _BLOCK   block to add to pipeline (NOTE: must be a subclass of Block)
            _PARENT_UID   UID of the block it receieves input from; if receiving input from LSL stream pass in "RAW"
            ...         varargs, passed onto _BLOCK.__init__()
        outputs:
            block_uid   The UID of the block added to the pipeline. (NOTE: Save this! Currently no way to retrieve it besides saving return value of add_block)

    select_source():
        allows you to interactively choose an input stream (NOTE: Only searches for EEG streams)
    run():
        run pipeline (background, multithreaded)
    stop():
        stop execution


Block:
    __init__():
        inputs:
            _pipe:  pipeline
            ...:    varargs, block-dependent
        outputs: none
        
        parameters passed to add_block are passed onto init. It is up to the block to type check all inputs

    run():
        inputs:
            buf:    input data; shape/type specific to each block
        outputs:
            dict(): dictionary of k,v pairs where k is the block_output_key and v is the actual data
    
    get_output_struct():
        inputs:
            none
        outputs:
            dict(): dictionary of k,v pairs where k is block_output_key and v is output length

    TODO: how to tell what data to pass to next block? pass whole dictionary, just values, what???


Block_LSL
    __init__():
        inputs:
            _pipe               pipeline (automatically passed by add_block)
            _parent_UID         UID of block whose output consitutes LSL stream output
            _parent_output_key  The output key of the parent block specifying which value in the dictionary should be output over LSL
            stream_name         The LSL stream name
            stream_type='EEG'   (Optional) LSL stream type
            stream_fs=250       (Optional) LSL stream FS
        outputs:
            none
            
    run():
        inputs:
            buf                 Data from parent block, buf[_parent_output_key] is broadcast over network w/ LSL
        outputs:
            none
            
    get_output_struct():
        inputs:
            none
        outputs:
            {}

'''




class Pipeline:
    '''
    High level pipeline class for Real-time EEG DSP
    '''
    
    def __init__(self, _BUF_LEN_SECS=20, _CHAN_SEL=[0], _SAMPLE_UPDATE_INTERVAL=4):
        self.mBufLenSecs = _BUF_LEN_SECS
        self.mSampleUpdateInterval = _SAMPLE_UPDATE_INTERVAL
        self.mChanSel = _CHAN_SEL
        self.mBlocks = dict() # Signal processing blocks are primary data transformation operations
        self.mInQueue = Queue() # Local fifo queue, usage: .put(), .get()
        self.mRunThreadEvent = Event()
        self.mRunThreadEvent.clear()
        
        #DBG
        self.mExecutionTimes=list()
        
    def get_self(self):
        return self
    
    def add_block(self, _BLOCK, _PARENT_UID, *args, **kwargs):
        '''
        Add signal processing block with required initialization args.
        Ensuring type correctness of init args should be handled in _BLOCK.

        returns block UID
        
        Example usage: pipeline.add_block(test_block, ['1','2'],{'kw1':'3','kw2':'4'})
        '''

        #Verify that _PARENT_UID is either present or we're using raw data
        if _PARENT_UID != "RAW" and _PARENT_UID not in self.mBlocks.keys():
             raise KeyError("_PARENT_UID does not exist in block list" + _PARENT_UID)
        
        #Verify that _BLOCK is valid
        if issubclass(_BLOCK, Block_LSL):
            block_function = _BLOCK(self, _PARENT_UID, *args, **kwargs)
        elif issubclass(_BLOCK, Block):
            block_function = _BLOCK(self, *args, **kwargs)
        else:
            raise TypeError("Input _BLOCK object does not inherit from Block class")
            
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
        
    def select_source(self):
        streams = resolve_stream()
        for i,s in enumerate(streams):
            print(i,s.name())
        stream_id = input("Input desired stream id: ")
        inlet = StreamInlet(streams[int(stream_id)])
        self.set_source(inlet)
    
    def set_source(self,_INLET):
        '''
        define intlet of type pylsl.StreamInlet
        usage: sample, timestamp = inlet.pull_sample()
        '''
        if isinstance(_INLET, pylsl.StreamInlet):
            self.mInlet = _INLET
            self.mInbufLen = int(self.mInlet.info().nominal_srate() * self.mBufLenSecs)
        else:
            raise TypeError("Requires type: "+ str(pylsl.StreamInlet) + ". Received type: "+ str(type(_INLET)))
            
    def _execute_block(self, _BLOCK_UID, _BUF):
        '''
        This method recurses through available blocks
        '''
        buf = self.mBlocks[_BLOCK_UID]['func'].run(_BUF)
        for next_block_uid, next_block_attrib in self.mBlocks.items():
            if next_block_attrib['parent'] == _BLOCK_UID:
                self._execute_block(next_block_uid, buf)
        return
            
    
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
    
    
    
    
    def acquisition_thread(self):
        '''
        Acquires samples from source and move them to local queue at fixed rate.
        Functions timing buffer to prevent execution thread from consuming all 
        input data and resulting in skipping.
        '''
        delay = 1.0/self.mInlet.info().nominal_srate()
        while self.mRunThreadEvent.is_set():
            data,timestamp = self.mInlet.pull_sample()
            self.mInQueue.put(data)
            time.sleep(delay)  
        
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
        buffer = deque(maxlen=self.mInbufLen) # buffer is (nsamples, nchan)
        while self.mRunThreadEvent.is_set():
            try:
                new_data = np.asarray(self.mInQueue.get(block=True, timeout=1.0/self.mInlet.info().nominal_srate()))
                buffer.append(new_data[self.mChanSel])
                new_count += 1
            except Empty:
                pass
            if new_count >= self.mSampleUpdateInterval and len(buffer) >= self.mInbufLen: 
                new_count = 0
                
                '''
                This is where actual thread execution occurs
                '''
                a=time.time()
                buffer_T = np.transpose(buffer) # buffer_T is nparray of shape (nchan, nsamples)
                for start_uid in starting_block_uids:
                    self._execute_block(start_uid, {'default':buffer_T})
                self.mExecutionTimes.append(time.time() - a)

    def run(self):
        '''
        Launch thread which executes signal processing pipeline until termination
        flag is set.
        '''
        if self.mRunThreadEvent.is_set():
            print("Run thread is already active")
        else:
            self.mRunThreadEvent.set()
            # Create threads
            acquisition_thread = Thread(target=self.acquisition_thread)
            run_thread = Thread(target=self.execution_thread)
            # Start threads
            acquisition_thread.start()
            run_thread.start()
        
    def stop(self):
        if not self.mRunThreadEvent.is_set():
            print("There is no active run thread to stop")
        else: 
            self.mRunThreadEvent.clear()