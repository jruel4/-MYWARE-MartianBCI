# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 07:33:14 2017

@author: MartianMartin
"""

from matplotlib import pyplot as plt

def plot_new_figure(x,y,id1,id2):
    fig = plt.figure()
    title = 'Stream: '+str(id1) +', Channel: '+str(id2)
    fig.suptitle(title, fontsize=20)
    ax = fig.add_subplot(1,1,1)
    ax.plot(x, y)
    plt.xlabel('Time (minutes)', fontsize=18)
    plt.ylabel('Power (microvolts)', fontsize=16)
    plt.show()
    
    