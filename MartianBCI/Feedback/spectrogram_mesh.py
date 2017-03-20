# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 02:24:01 2017

@author: marzipan
"""

import matplotlib.pyplot as plt
from time import sleep
from threading import Thread
from ..Utils.lsl_utils import select_stream
import numpy as np
import time

class visualize_spec_stream:
    
    def __init__(self, pipeline):
        # Wait for spec to be ready
        ctr = 0
        while True:
            if pipeline._blocks[0].spectrogram_ready():
                f,t = pipeline._blocks[0].get_axes() #TODO may have to wait for 20 second mark...
                break
            else:
                time.sleep(1)
                ctr += 1
                print("Waiting for spectrogram buffer to fill... "+str(ctr)+" seconds")
        
        print("Initializing graph")
        
        self.t = t
        self.f = f
        
        # select lsl stream
        self.inlet = select_stream()
        Sxx, timestamp = self.inlet.pull_sample()
        Sxx = np.asarray(Sxx).reshape(len(self.f),(len(self.t)))
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.qm = plt.pcolormesh(self.t, self.f, Sxx)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
        plt.ion()
        
        thread = Thread(target = self.visual_loop)
        thread.start()

    def visual_loop(self):
        try:
            while True:                
                Sxx, timestamp = self.inlet.pull_sample()                
                self.qm.set_array(np.asarray(Sxx))
                self.fig.canvas.draw()
                sleep(1.0/60.0)        
        except KeyboardInterrupt:
            pass

