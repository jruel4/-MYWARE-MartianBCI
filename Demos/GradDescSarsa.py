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

from RL_Environment_Simulator import RLEnv

#Clear graph, for tensorboard (and hygiene)
#tf.reset_default_graph()


##GLOBAL SETTINGS
Fs=250

## USER VARIABLES: FEEL FREE TO ADJUST ##
amp_min=0
amp_max = 1.0 #between 0 and 1.0, corresponds to computer max volume
amp_steps = 5
#amp_stepsize=0.1
amp = np.linspace(amp_min,amp_max,amp_steps)

bbf_min=1.0
bbf_max=19.0
bbf_steps=8
#bbf_stepsize=0.5
bbf = np.linspace(bbf_min, bbf_max, bbf_steps)

cf_min=250.0
cf_max=1000.0
cf_steps=5
#cf_stepsize=50.0
cf = np.linspace(cf_min, cf_max, cf_steps)

fbin_min = 0.5
fbin_max = 50.0
fbin_steps = 2
#fbin_stepsize = 0.25
fbins = np.concatenate(([0.0], np.linspace(fbin_min,fbin_max,fbin_steps), [Fs/2.0]))

_epsilon=0.1
_alpha=0.1
_gamma=0.1
_lambda=0.5



electrode_cnt = 8
feat_per_fbin_per_ch = 10
recording_len = 1000 #in samples

state_space_size = feat_per_fbin_per_ch*fbin_steps*electrode_cnt
act_space_size = amp_steps*bbf_steps*cf_steps
feat_space_size = state_space_size + act_space_size


### OBSOLETE

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






# in: x, length 3 tensor of amp x bbf x cf
def act_to_actbin_idx(x):
    #    print("x2: ",x)
    a=(x[0])
    b=(x[1]*amp_steps)
    c=(x[2]*amp_steps*bbf_steps)
    return (a+b+c)
#    return tf.constant(6000)

# in: x, length 3 tensor of amp x bbf x cf
def act_to_actbin(x):
#    print("x1: ",x)
    idx=act_to_actbin_idx(x)
    y = tf.SparseTensor([[idx]],[True],[act_space_size])
#    print('Y', y)
    return tf.sparse_tensor_to_dense(y,default_value=False)

def actbin_idx_to_act(idx):
    a=tf.floormod(idx,amp_steps)
    b=tf.floordiv(tf.floormod(idx,amp_steps*bbf_steps),bbf_steps)
    c=tf.floordiv(idx,(amp_steps*bbf_steps))
    return tf.stack([a,b,c],0)

def actbin_to_act(actbin):
    return actbin_idx_to_act(tf.argmax(actbin))

def greedy_action(input_data_new,w):
    act=tf.slice(w,[state_space_size],[-1])
    act=tf.arg_max(act,0)
    return actbin_idx_to_act(act)


def exploratory_action(data,p_action):
    arand= tf.random_uniform([1],minval=0,maxval=np.int64(amp_steps), dtype=tf.int64)
    brand= tf.random_uniform([1],minval=0,maxval=np.int64(bbf_steps), dtype=tf.int64)
    crand= tf.random_uniform([1],minval=0,maxval=np.int64(cf_steps), dtype=tf.int64)
    return tf.concat([arand, brand, crand], 0)

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
def generate_frequency_bins(L,Fs,bmin,bmax,bsteps):
    pphz=L/(Fs) #number of points per 1Hz
    binsize_hz=((bmax-bmin)/bsteps) #in Hz
    binsize_pnts=binsize_hz*pphz

    print("Point/Hz: ", pphz)
    print("Binsize Hz: ", binsize_hz)
    print("Binsize Points: ", binsize_pnts)

    #This is the mapping from FFT points returned to the frequency bins of interest
    bins = np.array([max(0,np.rint(pphz*bmin))]) #fft points we don't care about
    bins = np.append( bins, [np.rint(binsize_pnts) for x in range(bsteps)] ) #actual frequency bins of interest
    bins = np.append( bins, (L/2) - np.sum(bins)) #remaining points on one side of the fft
    bins = np.append( bins, (L/2) ) #other half of fft 
    return bins


##JCR 620
'''
Inputs:
    raw_input: #elec x #samples tensor of raw data
    f_start: start of the first frequency bin
    f_end: end of the last frequency bin
    num_bins: total number of bins
    
    (Optional) Fs: Frequency at which the data was sampled at; defaults to 250

Returns:
    out: #elec x num_bins tensor of average power across each frequency bin
'''
def extract_frequency_bins(raw_input,f_start,f_end,num_bins,Fs=250):
    with tf.name_scope("extract_fbins"):
        tf.assert_rank(raw_input,2,message="Error extracting frequency bins, input tensor must be rank 2, #elec x #samples.")
        
        assert( ((f_end-f_start) / float(num_bins)) / (Fs/L)
        
        L=raw_input.get_shape()[1]
        
        raw_fft=tf.fft(raw_input)

        #
        t_start=(L/Fs)*f_start
        t_end=(L/Fs)*f_end
        sliced_raw_fft = tf.slice(raw_fft,[0,t_start],[-1,t_end])
        
        
        sliced_raw_fft_t = 
        #tf.segment_sum (below) works on the first dimension of tensorflow
        #transpose the sensor so that we can use segment_sum below

##JCR 620

def map_next_action(data,p_action,w,e):
    with tf.name_scope("MapNextAction"):
        random = tf.random_uniform([1])
        next_action = tf.cond(random[0] > e,
                              lambda: greedy_action(data,w),
                              lambda: exploratory_action(data,p_action))
    return next_action

def map_reward(state,electrode_weights=[],num_elec=2,Fs=250,L=1000,window_overlap=25):

    bin_points=10    

    #generate the number of fft points per freq bin
    pp_fbin = generate_frequency_bins(L,Fs,8,13,bin_points)
    
    #generate a segment mapping for the tensor - like pp_fbin but each element in this array corresponds to a single element of the tensor
    segmap = generate_segment_map(pp_fbin,1)

    with tf.name_scope("Generate_Reward"):

        #generate fft
        spectro=tf.fft(tf.slice(state,[0,0],[-1, L]),name="raw_fft")
        
        #transpose the sensor so that we can use segment_sum below
        spectro=tf.transpose(spectro,name="pre_binning_transpose")
                
        #reduce the tensor using binning
        spectro_binned = tf.segment_sum(spectro,segmap,name="f_binning")
            
        #remove the first and last two bins, corresponding to <fbin_min, >fbin_max, and other fft half
        spectro_binned = spectro_binned[1:-2]
    
        #take fft power, transfer to new variable (no longer processing fft)
        reward = tf.abs(spectro_binned,"fft_abs")
        reward = tf.reshape(reward,[-1],name="flatten_fft")
        reward = tf.reduce_sum(reward,name="alphapow_summing")
        reward = tf.div(reward, bin_points,"mean")
    
    return reward
    
        #TODO Add in electrode weighting
        #    if electrode_weights == [] or len(electrode_weights) != num_elec:
        #        return tf.reduce_sum(spectro_binned[1:-2])
        #    else:
        #        e_w = tf.constant(electrode_weights)
        #        return tf.multiply(tf.reduce_sum(spectro_binned[1:-2]), electrode_weights)


## BINARY
def map_features_binary(state,action_,user_fbin_baseline,Fs=250,L=1000):
    #generate the number of fft points per freq bin
    pp_fbin = generate_frequency_bins(L,Fs,fbin_min,fbin_max,fbin_steps)
    
    #generate a segment mapping for the tensor - like pp_fbin but each element in this array corresponds to a single element of the tensor
    segmap = generate_segment_map(pp_fbin,1)

    with tf.name_scope("Map_Binary_Features"):

        #generate fft
        spectro=tf.fft(tf.slice(state,[0,0],[-1, L]),name="raw_fft")

        #transpose the sensor so that we can use segment_sum below
        spectro=tf.transpose(spectro,name="pre_binning_transpose")
        
        #reduce the tensor using binning
        spectro_binned = tf.segment_sum(spectro,segmap,name="f_binning")
    
        #remove the first and last two bins, corresponding to <fbin_min, >fbin_max, and other fft half
        spectro_binned = spectro_binned[1:-2]
        
        #make the tensor 
        spectro_binned = tf.transpose(spectro_binned,name="post_binning_transpose")

        #take fft power, transfer to new variable (no longer processing fft)
        state_features_tmp = tf.abs(spectro_binned)
        
        with tf.name_scope("Generate_Binary_State_Features"):
        
            #Extend this out so we can do a simple max with user baseline bins
            state_features_tmp = tf.tile(state_features_tmp,[1,feat_per_fbin_per_ch],name="expand_state_feat")
    
            #flatten the state features
            #NOTE: state_features_tmp should be #elec x #feat_per_fbin x #fbins
            state_features_tmp = tf.reshape(state_features_tmp,[-1,feat_per_fbin_per_ch,fbin_steps],name="mfb_reshape_state_feat")
        
            #TODO ensure theat GTE is right order on arguments
            GTE=tf.greater_equal(user_fbin_baseline,state_features_tmp)
            GTE=tf.cast(GTE,dtype=tf.int8)
            
            GTE_shifted=tf.pad(GTE, [[0,0],[1,0],[0,0]], mode='CONSTANT')
            GTE_shifted=tf.slice(GTE_shifted,[0,0,0],GTE.get_shape())
            state_features_bin=tf.not_equal(GTE,GTE_shifted,"isolate_tf")
            state_features_bin = tf.reshape(state_features_bin,[-1],name='flatten_state_space_tensor')
        
        with tf.name_scope("Generate_Binary_Action_Features"):
            action_features_bin = act_to_actbin(action_)
            action_features_bin = tf.reshape(action_features_bin,[-1],name='flatten_act_space_tensor')    
    
        features = tf.concat([state_features_bin,action_features_bin],0)
        return features

def load_test_data(fname='data001.xdf'):
    loaded_data=load_xdf(fname)
    time_series = loaded_data[0][0]['time_series']
    time_series=np.transpose(time_series)
    return time_series

def pull_raw_data_test():
    return

def pull_action_test():
    return

def load_individual_baseline_test():
    return

## CODE ##

def tf_init_model():
    #Create Variables
    with tf.variable_scope("root"):
        tf.get_variable(name="v_weights",shape=[feat_space_size],dtype=tf.float32,initializer=tf.constant_initializer(0))
        #tf.get_variable(name="v_weights",shape=[feat_space_size],dtype=tf.float32,initializer=tf.random_uniform_initializer(0,1))
        tf.get_variable(name="v_ztrace",shape=[feat_space_size],dtype=tf.float32,initializer=tf.constant_initializer(0))
        tf.get_variable(name="v_expected_reward",shape=[],dtype=tf.float32,initializer=tf.constant_initializer(0))

def tf_build_graph(tf_session,global_init,_user_fbin_baselines):
    with tf.variable_scope("root",reuse=True):
        
        # Graph Inputs
        raw_data_new = tf.placeholder(tf.complex64,shape=[1,1000],name='p_raw_data_new')

        # Constants
        epsilon = tf.constant(_epsilon, dtype=tf.float32,name='c_epsilon')
        alpha = tf.constant(_alpha, dtype=tf.float32,name='c_alpha')
        gamma = tf.constant(_gamma, dtype=tf.float32,name='c_gamma')
        lam = tf.constant(_lambda, dtype=tf.float32,name='c_lambda')
        user_fbin_baselines=tf.constant(_user_fbin_baselines,dtype=tf.float32,name='c_fbin_baselines')
    
        weights = tf.get_variable(name="v_weights",shape=[feat_space_size],dtype=tf.float32,initializer=tf.constant_initializer(0))
        z_trace = tf.get_variable(name="v_ztrace",shape=[feat_space_size],dtype=tf.float32,initializer=tf.constant_initializer(0))
        expected_reward = tf.get_variable(name="v_expected_reward",shape=[],dtype=tf.float32,initializer=tf.constant_initializer(0))
        
        #STEP 1: Calculate actual reward
        #TODO write unit tests for all sub routines
        actual_reward = map_reward(raw_data_new)
    
        #STEP 2: Find the error between expected and actual
        err_delta = tf.subtract(actual_reward, expected_reward, name="o_calc_err_delta")
    
        #STEP 3: Determine if we're greedy or not    
        #TODO eliminate exploratory behavior for initial testing
        action_next = map_next_action(raw_data_new,0,weights,epsilon)
    
        #STEP 5: Generate new S/A features
        bin_features = map_features_binary(raw_data_new,action_next,user_fbin_baselines)
        #Cast to float32 for easy operation
        bin_features = tf.cast(bin_features,dtype=tf.float32)
        
        #STEP 6: Generate new expected reward (used to update w,z,del)
        er = tf.reduce_sum(tf.multiply(weights,bin_features))
        expected_reward = er
    
        #STEP 7: Update our model
        err_delta = err_delta + gamma * expected_reward
        weights = weights + alpha * err_delta * z_trace
        z_tmp = (z_trace * gamma * lam) + bin_features #Decay, and update
        z_trace = tf.minimum(z_tmp,tf.ones(z_tmp.get_shape()),name="z_trace_replace") #uncomment to make replacing trace
    
        #STEP: Write summaries, return
        tf.summary.histogram("Weights",weights)
        tf.summary.histogram("EligibilityTrace",z_trace)
        tf.summary.histogram("ZTMP",z_tmp)
        tf.summary.histogram("BinFeat",bin_features)
        tf.summary.scalar("ExpectedReward",expected_reward)
        tf.summary.scalar("ER",er)
        tf.summary.scalar("ActualReward",actual_reward)
        tf.summary.scalar("ErrorDelta",err_delta)
        print(action_next)
        tf.summary.scalar("ActionNextA",action_next[0])
        tf.summary.scalar("ActionNextB",action_next[1])
        tf.summary.scalar("ActionNextC",action_next[2])
        summaries = tf.summary.merge_all()
        
    return raw_data_new,[action_next,summaries]

state=None
next_action=None
out=None
def main(tf_sess,global_init=True):    
    global state
    global out
    #
    individual_baselines=np.asarray(electrode_cnt*[np.transpose([range(500,5500,500)]*fbin_steps)])
    
    agent=RLEnv()
    
    tf_init_model()
    
    tf_in,tf_out = tf_build_graph(tf_sess,global_init,individual_baselines)

    init=tf.global_variables_initializer()

    # Setup summary writer with current graph
    if True: writer = tf.summary.FileWriter(".\\Logs\\",sess.graph)

    # Initialize all variables
    print("Initializing global variables...", sess.run(init))
    
    next_action=(bbf[0],amp[0])
    for i in range(10):
        try:
            state = agent.interact(next_action)
            out = sess.run(tf_out, feed_dict={tf_in: state})
            next_action=out[0]
            summaries=out[1]
            next_action=(bbf[next_action[1]],amp[next_action[0]])
            writer.add_summary(summaries, global_step=i)
        except KeyboardInterrupt:
            return
    return

tf.reset_default_graph()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    main(sess,global_init=True)
    sess.close()

if False:
    b=np.asarray(b)
    plt.plot(b[:,3])