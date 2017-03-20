# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 01:00:56 2017

@author: marzipan
"""

import time
import numpy as np
from pylsl import  StreamInlet, resolve_stream, StreamInfo, StreamOutlet
from threading import Thread

def create_test_source(freq=10, sps=250):
    '''
    create fake lsl stream of source data
    '''
    assert freq < (sps/2), "frequence must be less than nquist"
    stream_name = "Test_Signal_"+str(freq)+"_Hz_"
    stream_id = stream_name + time.strftime("_%d_%m_%Y_%H_%M_%S_")
    info = StreamInfo(stream_name, 'EEG', 8, 250, 'float32', stream_id)
    outlet = StreamOutlet(info)
    delay = 1.0/sps
    def _target():
        idx = 0
        mode = True
        while True:
            time.sleep(delay)
            idx += 1
            if idx % 2000 == 0:
                mode = not mode
            if mode:
               new_val = np.sin(2*np.pi*freq*(idx*delay))
            else:
                new_val = np.sin(2*np.pi*freq*2*(idx*delay))
            outlet.push_sample([new_val for i in range(8)])
    _thread = Thread(target=_target)
    _thread.start()
            
def select_stream():
    streams = resolve_stream('type', 'EEG')
    for i,s in enumerate(streams):
        print(i,s.name())
    stream_id = input("Input desired stream id: ")
    inlet = StreamInlet(streams[int(stream_id)])    
    return inlet




