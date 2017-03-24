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
from pylsl import  StreamInlet, resolve_stream, StreamInfo, StreamOutlet
from .Blocks.Block import Block

class Pipeline:
    '''
    High level pipeline class for Real-time EEG DSP
    '''
    
    def __init__(self, buf_len_secs=20, chan_sel=[0], sample_update_interval=4):
        self.buf_len_secs = buf_len_secs
        self.sample_update_interval = sample_update_interval
        self.chan_sel = chan_sel
        self._blocks = list() # Signal processing blocks are primary data transformation operations
        self._in_queue = Queue() # Local fifo queue, usage: .put(), .get()
        self.run_thread_event = Event()
        self.run_thread_event.clear()
        
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
            
    def select_output(self):
        stream_name = input("Enter desired outlet name: ")
        stream_id = stream_name + time.strftime("_%d_%m_%Y_%H_%M_%S_")
        info = StreamInfo(stream_name, 'EEG', self.get_output_len(), 60, 'float32', stream_id) 
        outlet = StreamOutlet(info)
        self.set_output(outlet)
        
    def set_output(self, outlet):
        '''
        define outlet of type pylsl.StreamOutlet
        usage: outlet.push_sample([0 for i in range(8)])
        '''
        if isinstance(outlet, pylsl.StreamOutlet):
            self._outlet = outlet
        else:
            raise TypeError("Requires type: "+ str(pylsl.StreamOutlet) + ". Received type: "+ str(type(outlet)))
        
    def get_output_len(self):
        last_block = self._blocks[-1]
        return last_block.get_output_dim(self.inbuf_len, self.chan_sel)
        
    def add_block(self, bFunction, *args, **kwargs):
        '''
        Add signal processing block with required initialization args.
        Ensuring type correctness of init args should be handled in bFunction.
        
        Example usage: pipeline.add_block(test_block, ['1','2'],{'kw1':'3','kw2':'4'})
        '''
        block_instance = bFunction(self, *args, **kwargs)
        assert isinstance(block_instance, Block)
        self._blocks.append(block_instance)
        
    def execute_blocks(self, buf):
        ''' 
        This method actuall passes data through chained block methods
        '''
        for block in self._blocks:
            buf = block.run(buf)
        return buf
        
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
                output = self.execute_blocks(buffer)
                # Output to lsl
                try:
                    self._outlet.push_sample(output)
                except ValueError as e:
                    print(e)
                    print("Received data.shape: "+str(output.shape))
        
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
        
        
    
    
    
    
    
    
    
    