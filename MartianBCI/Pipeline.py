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

class Pipeline:
    '''
    High level pipeline class for Real-time EEG DSP
    '''
    
    def __init__(self, buf_len_secs=20, chan_sel=[0], sample_update_interval=4):
        self.buf_len_secs = buf_len_secs
        self.sample_update_interval = sample_update_interval
        self.chan_sel = chan_sel
        self._blocks = dict() # Signal processing blocks are primary data transformation operations
        self._in_queue = Queue() # Local fifo queue, usage: .put(), .get()
        self.run_thread_event = Event()
        self.run_thread_event.clear()
    
    def add_block(self, bFunction, parentUID, *args, **kwargs):
        '''
        Add signal processing block with required initialization args.
        Ensuring type correctness of init args should be handled in bFunction.

        returns block UID
        
        Example usage: pipeline.add_block(test_block, ['1','2'],{'kw1':'3','kw2':'4'})
        '''

        #Verify that parentUID is either present or we're using raw data
        if parentUID != "RAW" and parentUID not in self._blocks.keys():
             raise KeyError("parentUID does not exist in block list" + parentUID)
        
        #Verify that bFunction is valid
        if issubclass(bFunction, Block_LSL):
            block_function = bFunction(self, parentUID, *args, **kwargs)
        elif issubclass(bFunction, Block):
            block_function = bFunction(self, *args, **kwargs)
        else:
            raise TypeError("Input bFunction object does not inherit from Block class")
            
        #Generate UID, verify that it doesn't already exist
        block_uid = np.random.randint(128000,256000)
        while block_uid in self._blocks.keys(): block_uid = np.random.randint(128000,256000)
        
        #Generate structure, prepend to blocklist
        new_block = {
                block_uid:{
                        'func':block_function,
                        'parent':parentUID,
                        }
                }
        self._blocks.update(new_block)

        return block_uid
        
    def select_source(self):
        streams = resolve_stream('type', 'EEG')
        for i,s in enumerate(streams):
            print(i,s.name())
        stream_id = input("Input desired stream id: ")
        inlet = StreamInlet(streams[int(stream_id)])
        self.set_source(inlet)
    
    def set_source(self,inlet):
        '''
        define intlet of type pylsl.StreamInlet
        usage: sample, timestamp = inlet.pull_sample()
        '''
        if isinstance(inlet, pylsl.StreamInlet):
            self._inlet = inlet
            self.inbuf_len = int(self._inlet.info().nominal_srate() * self.buf_len_secs)
        else:
            raise TypeError("Requires type: "+ str(pylsl.StreamInlet) + ". Received type: "+ str(type(inlet)))
            
    def execute_block(self, block_uid, _buf):
        '''
        This method recurses through available blocks
        '''
        buf = self._blocks[block_uid]['func'].run(_buf)
        for next_block_uid, next_block_attrib in self._blocks.items():
            if next_block_attrib['parent'] == block_uid:
                self.execute_block(next_block_uid, buf)
        return
            
    def acquisition_thread(self):
        '''
        Acquires samples from source and move them to local queue at fixed rate.
        Functions timing buffer to prevent execution thread from consuming all 
        input data and resulting in skipping.
        '''
        delay = 1.0/self._inlet.info().nominal_srate()
        while self.run_thread_event.is_set():
            data,timestamp = self._inlet.pull_sample()
            self._in_queue.put(data)
            time.sleep(delay)  
        
    def execution_thread(self):
        '''
        Executes signal processing pipeline
        '''
        
        starting_block_uids = list()
        for block_uid,v in self._blocks.items():
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