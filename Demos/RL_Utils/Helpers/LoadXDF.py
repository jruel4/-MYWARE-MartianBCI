# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt


from Demos.RL_Utils.Helpers.xdf import load_xdf



def load_test_data(fname='data001.xdf'):
    loaded_data=load_xdf(fname,handle_clock_resets=False,dejitter_timestamps=True)
    return loaded_data
    #d[0] is all applicable data, d[1] is just version info
    #d[0][x] is info for a stream
    #d[0][x]['time_series'] contains TS
    #d[0][x]['time_stamps']
    #d[0][x]['info']['name']
    
    ##NOTE ['time_stamps'] are in seconds


def plot_fft(channel,file='C:\Conda\MartianBCI\Demos\RL_Utils\Helpers\Recordings\JCR_IAF_06-22-17.xdf'):

    raw = load_test_data(file)
    
    #extract actual streams, and the marker streams
    streams=list()
    markers=list()
    for d in raw[0]:
        if 'AS' in d['info']['name'][0]:
            streams.append(d)
        elif 'Markers' in d['info']['type'][0]:
            markers.append(d)

    full_time_series = np.concatenate((streams[1]['time_series'],streams[0]['time_series'][0:len(streams[1]['time_series'])]),axis=1)
    chan_locs = ['F7','Fz','F8','C3','C4','Pz','O1','O2','Fp1','Fp2','F3','F4','Cz','P3','P4']

    
    L=len(streams[1]['time_series'])
    Fs=int(streams[1]['info']['nominal_srate'][0])
    start_ts = markers[0]['time_stamps'][6]
    end_ts = markers[0]['time_stamps'][6] + (1.0/Fs) * L

    correct_time_stamps = np.linspace(start_ts, end_ts, L)
    correct_marker_time_stamps=markers[0]['time_stamps'][6:]

    full_time_series = full_time_series[37500:93000,:]

    #for i in range(16):
    #    plt.plot(range(len(full_time_series[:,0])), full_time_series[:,i])
    #plt.show()
    
    O1 = full_time_series[:,channel]    
    L=len(O1)
    print(O1.shape)
    fft=np.fft.fft(O1)
    fft=fft
    Fs=250.0
    pphz=int(L/250)
    x_val=np.linspace(0,Fs,L)
    print(fft.shape)
    print(pphz)
    plt.plot(x_val[pphz*8:-(pphz*230)],np.abs(fft)[pphz*8:-(pphz*230)])


def get_raw_data(file='C:\Conda\MartianBCI\Demos\RL_Utils\Helpers\Recordings\JCR_IAF_06-22-17.xdf'):

    raw = load_test_data(file)
    
    #extract actual streams, and the marker streams
    streams=list()
    markers=list()
    for d in raw[0]:
        if 'AS' in d['info']['name'][0]:
            streams.append(d)
        elif 'Markers' in d['info']['type'][0]:
            markers.append(d)

    full_time_series = np.concatenate((streams[0]['time_series'][0:len(streams[1]['time_series'])], streams[1]['time_series']),axis=1)
    chan_locs = ['F7','Fz','F8','C3','C4','Pz','O1','O2','Fp1','Fp2','F3','F4','Cz','P3','P4']

    full_time_series = full_time_series[37500:93000,:]

    return [full_time_series,chan_locs]