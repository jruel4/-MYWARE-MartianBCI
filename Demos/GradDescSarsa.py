# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 12:43:38 2017

@author: marzipan
"""

#ROSHI uses phase of signals
#Different signals for each hemisphere

#Minimum delay is 150ms-250ms

###SMR
#The argument goes as follows: A narrow-band filter
#can be seen for our purposes as a transducer of frequency fluctuations into
#amplitude fluctuations. Frequency variation and phase fluctuations are obviously
#directly related. Dynamic, continuous reward-based training using narrow-band
#filters attempts to shape the EEG frequency distribution toward the middle of the
#resonance curve, with often immediate and sometimes trenchant consequences for
#the person’s state. These factors are in play even in ostensibly single-site amplitude
#training with referential placement (because references are not silent). 

#Bipolar montage then further augments the role of narrow-band filters as phase
#discriminants because the amplitudes at the two sites are now more correlated than in
#referential montage, which shifts the burden of variability more onto the phase. In
#typical application, the bipolar montage will be deployed either at near-neighbor
#sites or at homotopic sites. In these cases, the correlation of amplitudes (i.e.,
#comodulation) is typically enhanced with respect to arbitrary site combinations. 

                                                                             
###ALPHA
#In Les Fehmi’s mechanization of synchrony training, the reinforcement is
#delivered with every cycle of the alpha rhythm that meets criterion. It turns out
#that the timing of the delivery of the reward signal with respect to the underlying
#alpha signal is crucial. With the phase delay optimized, the reward pulse serves to
#augment the next cycle of the alpha spindle. This is firstly another demonstration
#that “phase matters. ” Secondly, we have here a stimulation aspect to what is fundamentally
#a feedback paradigm. 
                                                                             
#Record and *TAG* all data - we can look for correlations offline



import tensorflow as tf
import time
import numpy as np
from matplotlib import pyplot as plt

## USER VARIABLES: FEEL FREE TO ADJUST ##
amp_min=0
amp_max = 1.0 #between 0 and 1.0, corresponds to computer max volume
amp_steps = 10
#amp_stepsize=0.1
amp = np.linspace(amp_min,amp_max,amp_steps)

bbf_min=1.0
bbf_max=32.0
bbf_steps=32
#bbf_stepsize=0.5
bbf = np.linspace(bbf_min, bbf_max, bbf_steps)

cf_min=250.0
cf_max=1000.0
cf_steps=20
#cf_stepsize=50.0
cf = np.linspace(cf_min, cf_max, cf_steps)

fbin_min = 5.0
fbin_max = 20.0
fbin_steps = 30
#fbin_stepsize = 0.25

e_greedy=tf.constant([0.001], dtype=tf.float32)
#alpha=?
#gamma=?
#lam=?

fbins = np.linspace(fbin_min,fbin_max,fbin_steps)
fbins = np.insert(fbins,0,0)
fbins = np.insert(fbins,fbin_steps + 1,125)

#state_space = [[0]*2500]*21
#action_space = [[x,y,z] for x in np.asarray(amp) for y in np.asarray(bbf) for z in np.asarray(cf)]

Sin_1_5_10=[[np.sin((x/20)*2*np.pi)*2 + np.sin((x/25)*2*np.pi) + np.sin((x/250)*2*np.pi)*4 for x in range(0,10000)] * 2]

def greedy_action(input_data_new):
    #initialize variables
    next_action={'amp':amp[0],'bbf':bbf[0],'cf':cf[0]}
    next_features=map_features(input_data_new,next_action)
    next_features_tmp=next_features
    for next_amp in amp:
        for next_bbf in bbf:
            for next_cf in cf:
                next_features_tmp['action_features'] = map_features([], {'amp':next_amp,'bbf':next_bbf,'cf':next_cf})
                #Update best features and best action
                (next_features,next_action) = tf.case([tf.greater(next_features_tmp*w, next_features*w), lambda: (next_features_tmp,{'amp':amp[0],'bbf':bbf[0],'cf':cf[0]})],default=lambda: (next_features,next_action))
    return next_action


def exploring_action(current_action):
    amp_exp=min(amp_steps, max(0, (amp.index(current_action['amp']) + np.round(np.random.normal())))
    bbf_exp=min(bbf_steps, max(0, (bbf.index(current_action['bbf']) + np.round(np.random.normal())))
    cf_exp=min(cf_steps, max(0, (cf.index(current_action['cf']) + np.round(np.random.normal())))
    return {'amp':amp[amp_exp],'bbf':bbf[bbf_exp],'cf':cf[cf_exp]}

def generate_segment_map(bins,num_elec):
    x=list()
    for idx in range(len(bins)):
        x = x + [idx] * bins[idx].astype(int)
    if num_elec > 1:
        x = [x] * num_elec
    return np.transpose(x)

#generate array, each element corresponds to the number of FFT points
# to be summed for a given sample length, sampling frequency, and bin
#TODO: Add checking for valid values
def generate_frequency_bins(L,Fs,bmax, bmin, bsteps):
    pphz=L/(Fs) #number of points per 1Hz
    binsize_hz=((bmax-bmin)/bsteps) #in Hz
    binsize_pnts=binsize_hz*pphz
    #This is the mapping from FFT points returned to the frequency bins of interest
    bins = np.array([max(0,np.rint(pphz*(bmin - binsize_hz/2)))]) #fft points we don't care about
    bins = np.append( bins, [np.rint(binsize_pnts) for x in range(bsteps)] ) #actual frequency bins of interest
    bins = np.append( bins, (L/2) - np.sum(bins)) #remaining points on one side of the fft
    bins = np.append( bins, (L/2) ) #other half of fft 
    return bins

def map_reward(state,action,electrode_weights=[],num_elec=2,Fs=250,window_len=500,window_overlap=25):
    #generate the bins (number of points per bin)
    bins = generate_frequency_bins(window_len,250,13,8,10)
    
    #here we generate a segment mapping for the tensor
    segmap = generate_segment_map(bins,1)
    
    #here, we generate ffts for each window, take the absolute value, and store them as a #total_windows x #samples/window
    spectro=tf.fft(tf.slice(state,[0,0],[num_elec, window_len]))
        
    #reduce the tensor using binning
    spectro_binned = tf.segment_sum(tf.transpose(spectro),segmap)
    
    if electrode_weights.isempty() or len(electrode_weights) != num_elec:
        return tf.reduce_sum(spectro_binned[1:-2])
    else:
        e_w = tf.constant(electrode_weights)
        return tf.multiply(tf.reduce_sum(spectro_binned[1:-2]), electrode_weights)
    
def map_features(state,action,actual_reward,num_elec=2,Fs=250,window_len=500,window_overlap=25):
    #generate the bins (number of points per bin)
    bins = generate_frequency_bins(window_len,250,fbin_max,fbin_min,fbin_steps)
    
    #here we generate a segment mapping for the tensor
    segmap = generate_segment_map(bins,1)

    #here, we generate ffts for each window, take the absolute value, and store them as a #total_windows x #samples/window
    spectro=tf.fft(tf.slice(state,[0,0],[num_elec, window_len]))
        
    #reduce the tensor using binning
    spectro_binned = tf.segment_sum(tf.transpose(spectro),segmap)

    features = {'pow_'+str(fbins[i+1]): tf.abs(spectro_binned[i+1]) for i in range(len(fbins)-2)}

    labels=tf.constant(actual_reward)
    
    return features,labels

def load_test_data(fname='data001.xdf'):
    loaded_data=load_xdf(fname)
    time_series = loaded_data[0][0]['time_series']
    time_series=np.transpose(time_series)
    return time_series

def test_plot_fbins(ts_data,Fs=250):
    L=len(ts_data[0])
    print(L)
    pphz=L/(Fs) #number of points per 1Hz
    binsize_hz=((fbin_max-fbin_min)/fbin_steps) #in Hz
    binsize_pnts=binsize_hz*pphz
    #This is the mapping from FFT points returned to the frequency bins of interest
    bins = np.array([max(0,np.rint(pphz*(fbin_min - binsize_hz/2)))]) #fft points we don't care about
    bins = np.append( bins, [np.rint(binsize_pnts) for x in range(fbin_steps)] ) #actual frequency bins of interest
    bins = np.append( bins, (L/2) - np.sum(bins)) #remaining points on one side of the fft
    bins = np.append( bins, (L/2) ) #other half of fft

    features = list()
    for i in ts_data:
        m_fft = tf.fft(i)
        print(m_fft)
        print(bins)
        m_fft = tf.split(m_fft, bins.astype(int))
        features.append(m_fft)
    p=sess.run(features)
    for j in range(len(p)):
        dic1 = plt.figure()
        plt.plot(fbins,[np.abs(np.sum(p[j][i][:]))/len(p[j][i][:]) for i in range(len(p[j])-1)])

        dic2 = plt.figure()
        plt.plot(np.linspace(0,125,L/2),np.flip(np.abs(p[j][-1]),0))

        dic3 = plt.figure()
        plt.bar(range(len(fbins)),[np.abs(np.sum(p[j][i][:]))/len(p[j][i][:]) for i in range(len(p[j])-1)])
        plt.xticks(range(len(fbins)), fbins)
    
    return p

## CODE ##
writer = tf.summary.FileWriter(".\\Logs\\",sess.graph)
def main():

    #initialize variables
    weights = tf.Variable(tf.zeros(feature_space_size),dtype=tf.float32,name="weights")
    z_trace = tf.Variable(tf.zeros(feature_space_size),dtype=tf.float32,name="eligibility_trace")
    current_features = tf.Variable(tf.zeros(feature_space_size),dtype=tf.float32,name="SA_Pair")
    next_features = tf.Variable(tf.zeros(feature_space_size),dtype=tf.float32,name="Next_SA_Pair")
    
    #TODO - dimensions MxN or NxM?
    raw_data_new = tf.placeholder(tf.float32,name='input_data_old')
    raw_data_old = tf.placeholder(tf.float32,name='input_data_new')
    action_taken = tf.placeholder(tf.float32,name='action_taken') #this is the action taken which took us from old_data to new_data
    actual_reward = tf.placeholder(tf.float32,name='new_reward')
    
    features = [tf.contrib.layers.real_valued_column(column_name='stddev_'+str(i)) for i in fbins[1:-2]]
    features = [tf.contrib.layers.real_valued_column(column_name='mean_'+str(i)) for i in fbins[1:-2]]
    features = [tf.contrib.layers.real_valued_column(column_name='diff_'+str(i)) for i in fbins[1:-2]]
    features = [tf.contrib.layers.bucketized_column(i, boundaries=[-2 + j*0.25 for j in range(17)]) for i in features]

    def train_input_fn():
        return map_features(raw_data_old,action_taken,actual_reward)
    
    #TODO: Add queue for input data
    def live_input_fn():
        return map_features(raw_data_old,action_taken,actual_reward)
    
    #Generate actual reward
    actual_reward = generate_reward(raw_data_new)
    
    #STEP 2: Update eligibility traces
    z = tf.add(z,current_features,"Update ETrace Cum") #Cumulative trace
    #z = tf.minimum(z,tf.ones(feature_space_size)) #uncomment to make replacing trace
    
    #STEP 3: Find the error between expected and actual
    #Generate expected reward from old data's features
    expected_reward = tf.reduce_sum(tf.multiply(weights,current_features)) #b/c features old is binary this results in a summation of all weights for any features present
    
    #calculate error between expected and actual
    err_delta = actual_reward - expected_reward
    
    #STEP 5: Determine if we're greedy or not
    #STEP 5-1: Determine best greedy action by sweeping over action space
    #STEP 5-2: Explore
    random = tf.Variable(tf.random_uniform([1]), name="random_prob")
    next_action = tf.cond(tf.greater_equal(random_prob, epsilon), greedy_action, exploratory_action)
    
    #STEP 6: Generate new S/A features and expected reward, update w,z,del
    next_features = map_features(raw_data_new,next_action)
    next_expected_reward = tf.reduce_sum(tf.multiply(weights,next_features))
    
    err_delta = err_delta + gamma * next_expected_reward
    weights = weights + alpha * err_delta * z_trace
    z_trace = z_trace * gamma * lam
    
    #take actions
    updateAudioOut()
    delay()


    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # Runs the op.
    print(sess.run(c))