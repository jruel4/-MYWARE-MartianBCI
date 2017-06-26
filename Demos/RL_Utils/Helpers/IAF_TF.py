# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from Demos.RL_Utils.Helpers.LoadXDF import get_raw_data
from Demos.RL_Utils.Helpers import Processing_TF as Utils

fir_coeffs = '7-14_BP_FIR_BLACKMANN.npy'
#b,a = np.load('C:\\Conda\\MartianBCI\\Demos\\RL_Utils\\Filters\\' + fir_coeffs)
b,a = np.load('C:\\Users\\marzipan\\workspace\\MartianBCI\\Demos\\RL_Utils\\Filters\\' + fir_coeffs)

#rdd=get_raw_data()[0]
rdd=get_raw_data('C:\\Users\\marzipan\\workspace\\MartianBCI\\Demos\\RL_Utils\\Helpers\\Recordings\\JCR_IAF_06-22-17.xdf')[0]
#rd=rdd[2500:5000]
rd=rdd




def SinHz(hz,Fs=250,L=1000):
    return [np.sin(2*np.pi* x * hz / Fs) for x in range(L)]


#def detect_iaf(raw_input,)

if False:
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        
        raw_data_tensor = tf.constant( np.asarray(np.transpose(rd)), dtype=tf.float32)
        b_coeffs = tf.constant( b, dtype=tf.float32)

        data_bp_filt_tmp=Utils.multi_ch_conv(raw_data_tensor,b_coeffs)
        data_bp_filt=tf.slice(data_bp_filt_tmp, [0,300], [-1,tf.shape(data_bp_filt_tmp)[1]-600])

        pphz = tf.realdiv( tf.cast(tf.shape(data_bp_filt)[1], tf.float32) , tf.constant(250.0))

        #This value (in Hz) is used to determine the peak frequency - larger window uses more surrounding values to calculate IAF
        IAF_Peak_Window_Size = tf.constant(0.25 / 2.0)
        IAF_Window = tf.ones([tf.cast(tf.multiply(IAF_Peak_Window_Size,pphz),tf.int32)])

        data_fft = tf.fft(tf.cast(data_bp_filt,tf.complex64))
#        data_fft = fft_cpu([data_bp_filt[1]],1,2500)
        
        windowd = Utils.multi_ch_conv(tf.cast(tf.abs(data_fft),tf.float32),IAF_Window)
        
        init = tf.global_variables_initializer()
        print("Global variables init, ret: ", sess.run(init))
        writer = tf.summary.FileWriter(".\\Logs\\",sess.graph)
        e=sess.run([data_bp_filt,data_bp_filt_tmp,data_fft,windowd])








'''
Input:
    raw_input: #chan x #samples tensor
    w_len: length of window
    w_overlap: overlap between windows


'''
def specgram_tf_2d(raw_input, w_len, w_overlap, window=[]):

    #Get input signal length
    s_len = tf.shape(raw_input)[1]
    n_chan = tf.shape(raw_input)[0]

    #Determine how much we shift the window by
    w_shift = w_len - w_overlap
    
    window = signal.tukey(w_len)
    window_tf = tf.pad( tf.constant(window,dtype=tf.float32), [[0,(s_len - w_len)]], mode='CONSTANT')
    window_tf = tf.expand_dims(window_tf,0)
    w_var = tf.tile(window_tf,[n_chan,1])
    #w_var = window_tf

    specgram_len = tf.floordiv((s_len - w_len), w_shift)



    #Zero-mean the data    
#    mean = tf.reduce_mean(raw_input[0])
#    s_var = tf.subtract(raw_input[0],mean)
    s_var = raw_input
    
    #Initialize our loop variables
    idx = tf.constant(0,tf.int32)    
    specgram_init = tf.zeros(shape=[specgram_len, n_chan, s_len], dtype=tf.complex64 )
    loopvars = [idx,specgram_init]
    
    def bod(l_idx,specgram):
        w_tmp = Utils.shift_2d(w_var, tf.cast(w_shift * l_idx,tf.int32),1)
        s_tmp = tf.multiply(s_var, w_tmp)
        fft_tmp = tf.fft(tf.cast(s_tmp,tf.complex64))
        fft_padded = tf.pad(tf.expand_dims(fft_tmp,0),[[l_idx, tf.shape(specgram)[0] - l_idx - 1],[0,0],[0,0]])
        
        #Update loop variables
        l_idx = l_idx + 1
        specgram=tf.add(specgram, fft_padded)
    
        return [l_idx,specgram]
    
    # total_len - window_len  > (overlap*(idx))
    def cond(ll_idx,lcl_r_var):
        return tf.greater( s_len - w_len, tf.cast(w_shift * ll_idx, tf.int32 ))
        
    wh = tf.while_loop(cond,bod,loopvars,parallel_iterations=100000)
    return wh





##SPECGRAM TST
if True:
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        raw_data_tensor = tf.constant( np.asarray(np.transpose(rd)), dtype=tf.float32)
        
        b_coeffs = tf.constant( b, dtype=tf.float32)

        data_bp_filt_tmp=Utils.multi_ch_conv(raw_data_tensor,b_coeffs)
        data_bp_filt=tf.slice(data_bp_filt_tmp, [0,300], [-1,tf.shape(data_bp_filt_tmp)[1]-600])
        
        raw_data_tensor=data_bp_filt

        z=specgram_tf_2d(raw_data_tensor,500,450)
        init = tf.global_variables_initializer()
        print("Global variables init, ret: ", sess.run(init))
        writer = tf.summary.FileWriter(".\\Logs\\",sess.graph)
        global p
        p=sess.run([z])















'''
Input:
    raw_input: #chan x #samples tensor
    w_len: length of window
    w_overlap: overlap between windows


'''
def specgram_tf_1d(raw_input, w_len, w_overlap, window=[]):

    #Get input signal length
    s_len = tf.shape(raw_input)[1]

    #Determine how much we shift the window by
    w_shift = w_len - w_overlap
    
    window = signal.tukey(w_len)
    window_tf = tf.pad( tf.constant(window,dtype=tf.float32), [[0,(s_len - w_len)]], mode='CONSTANT')
    #    window_tf = tf.expand_dims(window_tf,0)
    #    w_var = tf.tile(window_tf,[tf.shape(raw_input)[0],1])
    w_var = window_tf

    specgram_len = tf.floordiv((s_len - w_len), w_shift)



    #Zero-mean the data    
    mean = tf.reduce_mean(raw_input[0])
    s_var = tf.subtract(raw_input[0],mean)

    #Initialize our loop variables
    idx = tf.constant(0,tf.int32)    
    specgram_init = tf.zeros(shape=[specgram_len, s_len], dtype=tf.complex64 )
    loopvars = [idx,specgram_init]
    
    def bod(l_idx,specgram):
        w_tmp = Utils.shift_1d(w_var, tf.cast(w_shift * l_idx,tf.int32))
        s_tmp = tf.multiply(s_var, w_tmp)
        fft_tmp = tf.fft(tf.cast(s_tmp,tf.complex64))
        fft_padded = tf.pad(tf.expand_dims(fft_tmp,0),[[l_idx, tf.shape(specgram)[0] - l_idx - 1],[0,0]])
        
        #Update loop variables
        l_idx = l_idx + 1
        specgram=tf.add(specgram, fft_padded)
    
        return [l_idx,specgram]
    
    # total_len - window_len  > (overlap*(idx))
    def cond(ll_idx,lcl_r_var):
        return tf.greater( s_len - w_len, tf.cast(w_shift * ll_idx, tf.int32 ))
        
    wh = tf.while_loop(cond,bod,loopvars,parallel_iterations=100000)
    return wh
    


def plot_spectro(raw_in):
    plt.pcolormesh(t,f[125:500],np.transpose(np.abs(p[0][1][:,125:500])))
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()


##SPECGRAM TST
if False:
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        raw_data_tensor = tf.constant( np.asarray(np.transpose(rd)), dtype=tf.float32)
        
        b_coeffs = tf.constant( b, dtype=tf.float32)

        data_bp_filt_tmp=Utils.multi_ch_conv(raw_data_tensor,b_coeffs)
        data_bp_filt=tf.slice(data_bp_filt_tmp, [0,300], [-1,tf.shape(data_bp_filt_tmp)[1]-600])
        
        raw_data_tensor=data_bp_filt

        z=specgram_tf_1d(raw_data_tensor,500,495)
        init = tf.global_variables_initializer()
        print("Global variables init, ret: ", sess.run(init))
        writer = tf.summary.FileWriter(".\\Logs\\",sess.graph)
        global p
        p=sess.run([z])
        

    


if False:
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        raw_data_tensor = tf.constant( np.asarray(np.transpose(rd)), dtype=tf.float32)
        
        b_coeffs = tf.constant( b, dtype=tf.float32)

        data_bp_filt_tmp=Utils.multi_ch_conv(raw_data_tensor,b_coeffs)
        data_bp_filt=tf.slice(data_bp_filt_tmp, [0,300], [-1,tf.shape(data_bp_filt_tmp)[1]-600])
        
        raw_data_tensor=data_bp_filt
        
        w_len = 500
        s_len = tf.shape(raw_data_tensor)[1]
        w_overlap = 50
        #Generate window matrix
        window = signal.tukey(w_len)
        window_tf = tf.pad( tf.constant(window,dtype=tf.float32), [[0,(s_len - w_len)]], mode='CONSTANT')
    #    window_tf = tf.expand_dims(window_tf,0)
    #    w_var = tf.tile(window_tf,[tf.shape(raw_data_tensor)[0],1])
        w_var = window_tf

        #Zero-mean the data    
        mean = tf.reduce_mean(raw_data_tensor[0])
        s_var = tf.subtract(raw_data_tensor[0],mean)

        idx = tf.constant(0,tf.int32)
        r_var = tf.zeros(
                    name="Spectro",
                    shape=[
                            tf.floordiv((s_len - w_len), w_overlap),
                            tf.shape(raw_data_tensor)[1]
                        ],
                    dtype=tf.complex64,
                )
#        tf.get_variable()
        loopvars = [idx,r_var]
        
        def bod(lcl_idx,lcl_r_var):
            w_tmp = Utils.shift_1d(w_var, tf.cast(w_overlap * lcl_idx,tf.int32))
            s_tmp = tf.multiply(s_var, w_tmp)
            fft_tmp = tf.fft(tf.cast(s_tmp,tf.complex64))
            fft_padded = tf.pad(tf.expand_dims(fft_tmp,0),[[lcl_idx, tf.shape(lcl_r_var)[0] - lcl_idx - 1],[0,0]])

            lcl_idx = lcl_idx + 1
            lcl_r_var=tf.add(lcl_r_var, fft_padded)

            return [lcl_idx,lcl_r_var]
        
        # total_len - window_len  > (overlap*(idx))
        def cond(lcl_idx,lcl_r_var):
            return tf.greater(
                    s_len - w_len,
                    tf.cast(w_overlap * lcl_idx, tf.int32)
                    )
            
        wh = tf.while_loop(cond,bod,loopvars,parallel_iterations=100000)
        
        init = tf.global_variables_initializer()
        print("Global variables init, ret: ", sess.run(init))
        writer = tf.summary.FileWriter(".\\Logs\\",sess.graph)
#        global p
        p=sess.run([wh])


if False:
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        def f1(): return tf.constant(1)
        def f2(): return tf.constant(2)
        a=tf.cond(tf.greater(1,10),f1,f2)
        c=sess.run(a)
        
        
        q = tf.get_variable('Q',shape=[2,2],dtype=tf.int32)
        ass=tf.assign(q,tf.constant([[1,1],[2,2]]))
        with tf.control_dependencies([ass]):
            x=tf.constant([[1,1],[2,2]])
            y=tf.scatter_add(q,[1],[[2,2]])
            z=sess.run(y)

#plt.plot(np.linspace(0,250,3050),np.abs(np.fft.fft(x[0]))[25:-25])

###SPECTRO
'''f,t,Sxx = signal.spectrogram(np.transpose(rdd)[1],250)




shift
multiply elementwise
fft
save_as_fft




rd_tran = np.asarray(np.transpose(rd))

filtd=signal.filtfilt(b,a,rd_tran[0])

f, t, Sxx = signal.spectrogram(filtd, 250,nperseg=500,noverlap=475)
plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()


plt.pcolormesh(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
'''

#L=1000



#b=np.transpose()
#b=SinHz(5)

#c=np.matmul(a,b)