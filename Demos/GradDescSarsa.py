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

#Clear graph, for tensorboard (and hygiene)
#tf.reset_default_graph()


##GLOBAL SETTINGS
Fs=250

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

fbin_min = 0.5
fbin_max = 50.0
fbin_steps = 100
#fbin_stepsize = 0.25
fbins = np.concatenate(([0.0], np.linspace(fbin_min,fbin_max,fbin_steps), [Fs/2.0]))

_epsilon=0.1
_alpha=0.1
_gamma=0.1
_lambda=0.9



electrode_cnt = 8
feat_per_fbin_per_ch = 10
recording_len = 1000 #in samples

state_space_size = feat_per_fbin_per_ch*fbin_steps*electrode_cnt
act_space_size = amp_steps*bbf_steps*cf_steps
feat_space_size = state_space_size + act_space_size



#state_space = [[0]*2500]*21
#action_space = [[x,y,z] for x in np.asarray(amp) for y in np.asarray(bbf) for z in np.asarray(cf)]

Sin_1_2_4=[np.sin((x/250)*2*np.pi)*4 + np.sin((x/125)*2*np.pi)*2 + np.sin((x/62.5)*2*np.pi)*1 for x in range(0,10000)]
Sin_8_12_16=[np.sin((x/25)*2*np.pi)*4 + np.sin((x/20.833)*2*np.pi)*2 + np.sin((x/12.5)*2*np.pi)*1 for x in range(0,10000)]
Sin_5=[np.sin((x/50)*2*np.pi)*1 for x in range(0,10000)]
Sin_10=[np.sin((x/25)*2*np.pi)*1 for x in range(0,10000)]
Sin8CH=np.concatenate((([Sin_1_2_4] * 4),([Sin_8_12_16]* 4)),axis=0)
SINNR=[Sin_5]*8
SINR=[Sin_10]*8
_tSin8CH=tf.constant(Sin8CH,dtype=tf.complex64)

# in: x, length 3 tensor of amp x bbf x cf
def act_to_actbin_idx(x):
    #    print("x2: ",x)
    a=(x[0])
    b=(x[1]*amp_steps)
    c=(x[2]*amp_steps*bbf_steps)
    print(amp_steps)
    print(bbf_steps)
    print(a,b,c)
    return [x[0],x[1],x[2]],(a+b+c)
#    return tf.constant(6000)

# in: x, length 3 tensor of amp x bbf x cf
def act_to_actbin(x):
#    print("x1: ",x)
    p,idx=act_to_actbin_idx(x)
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

def test():
    return

def greedy_action(input_data_new,w):
    #initialize variables
#    print("GREEDY SLICE: ",tf.slice(w,[state_space_size],[-1]))
    act=tf.slice(w,[state_space_size],[-1])
    act=tf.arg_max(act,0)
#    print("GREEDY ARGMAX: ", act)
    print(actbin_idx_to_act(act))
#    return tf.cast(tf.constant([4,5,6]),dtype=tf.int64)
    return actbin_idx_to_act(act)
#    
#    next_features=map_features(input_data_new,next_action)
#    next_features_tmp=next_features
#    for next_amp in amp:
#        for next_bbf in bbf:
#            for next_cf in cf:
#                next_features_tmp['action_features'] = map_features([], {'amp':next_amp,'bbf':next_bbf,'cf':next_cf})
#                #Update best features and best action
#                (next_features,next_action) = tf.case([tf.greater(next_features_tmp*w, next_features*w), lambda: (next_features_tmp,{'amp':amp[0],'bbf':bbf[0],'cf':cf[0]})],default=lambda: (next_features,next_action))
#    return next_action

def exploratory_action(data,p_action):
    arand= tf.random_uniform([1],minval=0,maxval=np.int64(amp_steps), dtype=tf.int64)
    brand= tf.random_uniform([1],minval=0,maxval=np.int64(bbf_steps), dtype=tf.int64)
    crand= tf.random_uniform([1],minval=0,maxval=np.int64(cf_steps), dtype=tf.int64)
    return tf.concat([arand, brand, crand], 0)
#    return tf.cast(tf.constant([1,2,3]),dtype=tf.int64)

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


next_action = None
def map_next_action(data,p_action,w,e):
    global next_action
    random = tf.random_uniform([1])
    next_action = tf.cond(random[0] > e,
                          lambda: greedy_action(data,w),
                          lambda: exploratory_action(data,p_action),
                          name='greedy_or_random_condition')
    return next_action

def map_reward(state,electrode_weights=[],num_elec=2,Fs=250,L=1000,window_overlap=25):
    #generate the bins (number of points per bin)
    pp_fbin = generate_frequency_bins(L,Fs,8,13,10)
    
#    if True: print("Freq Bins: ", fbins)
#    if True: print("Bins: (PPBin)", pp_fbin)
    
    #generate a segment mapping for the tensor - like pp_fbin but each element in this array corresponds to a single element of the tensor
    segmap = generate_segment_map(pp_fbin,1)

#    if True: print("Shape of segmap: ", np.shape(segmap))

    #generate fft
    spectro=tf.fft(tf.slice(state,[0,0],[-1, L]),name="mrwd_fft")
    
#    if True: print("Spectro: ", spectro)
#    if True: print("Spectro transposed: ", tf.transpose(spectro))
    
    #reduce the tensor using binning
    #TODO: Check if it is necessary to transpose when using segment sum
    spectro_binned = tf.segment_sum(tf.transpose(spectro),segmap,name="mrwd_fbinning")

    #Make this easier to manipulate (and nix everything we don't care about)
    spectro_binned = tf.transpose(spectro_binned[1:-2]) #1 eliminates DC, -2 eliminates HF and opposite side of spectro

    # TODO - MAKE BINARY TENSOR
    
    reward = tf.div(tf.reduce_sum(tf.reshape(tf.abs(spectro_binned),[-1]),name="mrwd_alphapow_summing"), 10)
    return reward
    #TODO
#    if electrode_weights == [] or len(electrode_weights) != num_elec:
#        return tf.reduce_sum(spectro_binned[1:-2])
#    else:
#        e_w = tf.constant(electrode_weights)
#        return tf.multiply(tf.reduce_sum(spectro_binned[1:-2]), electrode_weights)


## BINARY
def map_features_binary(state,action_,user_fbin_baseline,Fs=250,L=2500):
    #generate the number of fft points per freq bin
    pp_fbin = generate_frequency_bins(L,Fs,fbin_min,fbin_max,fbin_steps)
    
#    if True: print("Freq Bins: ", fbins)
#    if True: print("Bins: (PPBin)", pp_fbin)
    
    #generate a segment mapping for the tensor - like pp_fbin but each element in this array corresponds to a single element of the tensor
    segmap = generate_segment_map(pp_fbin,1)

#    if True: print("Shape of segmap: ", np.shape(segmap))

    #generate fft
    spectro=tf.fft(tf.slice(state,[0,0],[-1, L]),name="mfb_fft")
    
#    if True: print("Spectro: ", spectro)
#    if True: print("Spectro transposed: ", tf.transpose(spectro))
    
    #reduce the tensor using binning
    #TODO: Check if it is necessary to transpose when using segment sum
    spectro_binned = tf.segment_sum(tf.transpose(spectro),segmap,name="mfb_fbinning")

    #Make this easier to manipulate (and nix everything we don't care about)
    state_features_tmp = tf.abs(tf.transpose(spectro_binned[1:-2],name="mfb_clean_bins")) #1 eliminates DC, -2 eliminates HF and opposite side of spectro

    # MAKE BINARY TENSOR
    #state_features_tmp  = tf.abs(spectro_binned)
    
    #Extend this out so we can do a simple max, then G/E op
    state_features_tmp = tf.tile(state_features_tmp,[1,feat_per_fbin_per_ch],name="mfb_expand_state_feat")
    state_features_tmp = tf.reshape(state_features_tmp,[-1,feat_per_fbin_per_ch,fbin_steps],name="mfb_reshape_state_feat")
    #state_features_tmp should be #elec x #feat_per_fbin x #fbins

    GTE=tf.greater_equal(user_fbin_baseline,state_features_tmp)
    GTE=tf.cast(GTE,dtype=tf.int8)
    
    GTE_shifted=tf.pad(GTE, [[0,0],[1,0],[0,0]], mode='CONSTANT')
    GTE_shifted=tf.slice(GTE_shifted,[0,0,0],GTE.get_shape())
    statebin=tf.not_equal(GTE,GTE_shifted,"mfb_generate_binary_feats")
#    for i in range(feat_per_fbin_per_ch):
#       user_fbin_baseline = tf.multiply(tf.cast(GTE,dtype=tf.float32),user_fbin_baseline)

    #IGNORE
    #NOTE: To have state_features_tmp should be #elec x #fbins x #feat_per_fbin, uncomment
#    state_features_tmp = tf.transpose()
#    state_features_tmp = tf.reshape(state_features_tmp,[-1,feat_per_fbin_per_ch,fbin_steps])

    
#    for i in 8:
#    state_features_tmp = 
#    tf.reduce_sum(tf.greater_equal(state_features_tmp, user_fbin_baseline))
    
#    if True: print("state_features_tmp:", state_features_tmp)

    actbin = act_to_actbin(action_)
    actbin = tf.reshape(actbin,[-1],name='flatten_act_space_tensor')    
    statebin = tf.reshape(statebin,[-1],name='flatten_state_space_tensor')
    return tf.concat([statebin,actbin],0) #padding here because we are not dicking around with action space

#####DEPRECATED    
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

def pull_raw_data_test():
    return

def pull_action_test():
    return

def load_individual_baseline_test():
    return

## CODE ##
def tf_init_graph(tf_session,global_init):
    #get user baseline power
    #individual_baselines = load_individual_baseline()
    
    
    ## TENSORFLOW BEGIN
    
    #TEST - BINS BETWEEN 1-10
    individual_baselines=tf.constant(np.asarray(electrode_cnt*[np.transpose([range(feat_per_fbin_per_ch)]*fbin_steps)]),dtype=tf.float32)
    
    
    ##TF CONSTANTS
    epsilon = tf.constant(_epsilon, dtype=tf.float32,name='c_epsilon')
    alpha = tf.constant(_alpha, dtype=tf.float32,name='c_alpha')
    gamma = tf.constant(_gamma, dtype=tf.float32,name='c_gamma')
    lam = tf.constant(_lambda, dtype=tf.float32,name='c_lambda')
    
    #TF VARIABLES
    weights = tf.get_variable(name="v_weights",shape=[feat_space_size],dtype=tf.float32)
    z_trace = tf.get_variable(name="v_z_trace",shape=[feat_space_size],dtype=tf.float32)
    expected_reward = tf.get_variable(name="v_expected_reward",shape=[],dtype=tf.float32)

    ##TF PLACEHOLDERS (INPUTS)
    raw_data_new = tf.placeholder(tf.complex64,name='p_raw_data_new')

#    action_taken_ovrrd = tf.placeholder(tf.float32,name='action_taken_override') #this is the action taken which took us from old_data to new_data
#    actual_reward_ovrrd = tf.placeholder(tf.float32,name='new_reward_override')
 
# This will be used when we set up an actual model   
#    features = [tf.contrib.layers.real_valued_column(column_name='stddev_'+str(i)) for i in fbins[1:-2]]
#    features = [tf.contrib.layers.real_valued_column(column_name='mean_'+str(i)) for i in fbins[1:-2]]
#    features = [tf.contrib.layers.real_valued_column(column_name='diff_'+str(i)) for i in fbins[1:-2]]
#    features = [tf.contrib.layers.bucketized_column(i, boundaries=[-2 + j*0.25 for j in range(17)]) for i in features]

#==============================================================================
#     def train_input_fn():
#         return map_features(raw_data_old,action_taken,actual_reward)
#     
#     #TODO: Add queue for input data
#     def live_input_fn():
#         return map_features(raw_data_old,action_taken,actual_reward)
#     
#==============================================================================
    
    #STEP 1: Update eligibility traces


    #STEP 2: Find the error between expected and actual
    #Generate expected reward from old data's features
    #b/c features old is binary this results in a summation of all weights for any features present
    

    #STEP 3: Calculate actual reward
    actual_reward = map_reward(raw_data_new)

    #calculate error between expected and actual
    err_delta = tf.subtract(actual_reward, expected_reward, name="calc_err_delta")

    #STEP 4: Determine if we're greedy or not
    #STEP 4-1: Determine best greedy action by sweeping over action space
    #STEP 4-2: Explore
    
    action_next = map_next_action(raw_data_new,0,weights,epsilon)

    #STEP 5: Generate new S/A features
    bin_features = tf.cast(map_features_binary(raw_data_new,action_next,individual_baselines),dtype=tf.float32)
    
    #STEP 6: Generate new expected reward (used to update w,z,del)
    expected_reward.assign(tf.reduce_sum(tf.multiply(weights,bin_features)))

    #STEP 7: Update our model
    err_delta = err_delta + gamma * expected_reward
    weights.assign(weights + alpha * err_delta * z_trace)
    z_trace.assign(z_trace * gamma * lam)
    
    z_trace.assign(tf.add(z_trace,bin_features,name="update_z_trace_accumulate")) #Cumulative trace
    z_trace.assign(tf.minimum(z_trace,tf.ones(z_trace.get_shape()),name="update_z_trace_replace")) #uncomment to make replacing trace

    if global_init:
        init = tf.global_variables_initializer()
        print("Global variables init", tf_session.run(init))
    
    return raw_data_new,[action_next,weights,z_trace,actual_reward,expected_reward]
    
def main(tf_sess,state,global_init=True):    
    tf_in,tf_out=tf_init_graph(tf_sess,global_init)
    next_a = list()
#    while True:
    for i in range(1000):
        
        Sin_5=[np.sin((x/50)*2*np.pi)*1 for x in range(0,10000)]
        Sin_10=[np.sin((x/25*i/25.0)*2*np.pi)*1 for x in range(0,10000)]
        Sin8CH=np.concatenate((([Sin_1_2_4] * 4),([Sin_8_12_16]* 4)),axis=0)
        SINNR=[Sin_10]*8
#        print(i)
#        state = env_interact()
        #TEST
#        action_new = pull_action_test()
        #Generate actual reward
#        actual_reward = pull_reward_test()
        # Pull new data from a stream
#        raw_data_new = pull_raw_data_test()
        try:
            next_a.append(sess.run(tf_out, feed_dict={tf_in:SINNR}))
            #TODO evaluate action not weights
        except KeyboardInterrupt:
            return
    return next_a


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
b=main(sess,SINR,global_init=True)
if True: writer = tf.summary.FileWriter(".\\Logs\\",sess.graph)
#sess.close()

if False:
    b=np.asarray(b)
    plt.plot(b[:,3])