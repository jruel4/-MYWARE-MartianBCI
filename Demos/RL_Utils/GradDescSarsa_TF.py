'''
Top level class for Gradient Descent Sarsa (Lambda) RL Algorithm

Usage: Input a 'state', output as 'action'. 

E.g. input eeg data, output audio commands.

TODO Entire tensorflow session shall be encapsulated by this class
'''

import tensorflow as tf
import numpy as np

import time

from Demos.RL_Utils.Helpers import ActionSelection_TF as ActionSelection
from Demos.RL_Utils.Helpers import BinaryFeatureExtraction_TF as BinaryFeatureExtractor
from Demos.RL_Utils.Helpers import RewardExtraction_TF as RewardExtraction
from Demos import RL_Environment_Simulator as RLEnv


class GradDescSarsaAgent:
    
    def __init__(self,_EPSILON=0.75,_ALPHA=0.9,_GAMMA=0,_LAMBDA=0,_ALPHA_DECAY=.999):

        
        self._EPSILON=_EPSILON
        self._ALPHA=_ALPHA
        self._GAMMA=_GAMMA
        self._LAMBDA=_LAMBDA
        self._ALPHA_DECAY = _ALPHA_DECAY

        self._init_class_var()
        self._init_tf_session()
        self._init_class_var_tf()        

        self.mFeatExtract = BinaryFeatureExtractor.BinaryFeatureExtractor([len(self.mBBF),1])
        self.mRewardExtract = RewardExtraction.RewardExtraction()
        self.mActSel = ActionSelection.ActionSelection(self.mFeatExtract,self.mRewardExtract,self.mAMP,self.mBBF,self.mCF)
        
        self.mFeatureSpaceSize = self.mFeatExtract.getBinaryFeatureSpaceSize()
        
        self.setup()
        
        #do everything
        
        return
    
    def _init_tf_session(self):
        self.mSess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        return

    def _init_class_var(self):
         
        self.mLOG_GRAPH = True
        
        ## USER VARIABLES: FEEL FREE TO ADJUST ##
        amp_min=0
        amp_max = 1.0 #between 0 and 1.0, corresponds to computer max volume
        amp_steps = 5
        #amp_stepsize=0.1
        self.mAMP = np.linspace(amp_min,amp_max,amp_steps)
        
        bbf_min=1
        bbf_max=50
        bbf_steps=50
        #bbf_stepsize=0.5
        self.mBBF = np.linspace(bbf_min, bbf_max, bbf_steps)
        
        cf_min=250.0
        cf_max=1000.0
        cf_steps=5
        #cf_stepsize=50.0
        self.mCF = np.linspace(cf_min, cf_max, cf_steps)
        
    def _init_class_var_tf(self):

        # Constants
        self.mEPSILON = tf.constant(self._EPSILON, dtype=tf.float32,name='EPSILON')
        self.mGAMMA = tf.constant(self._GAMMA, dtype=tf.float32,name='GAMMA')
        self.mLAMBDA = tf.constant(self._LAMBDA, dtype=tf.float32,name='LAMBDA')
        self.mALPHA_DECAY = tf.constant(self._ALPHA_DECAY, dtype=tf.float32,name='ALPHA_DECAY')

    def reset(self):
        #Resets graph (all nodes) - useful when runnning often
        tf.reset_default_graph()
        return
    
    def setup(self,log_graph=True):
        self.mGraphIn,self.mGraphOut = self.build_main_graph()
        #TODO: self.generate_summaries()

        # Initialize all variables
        init=tf.global_variables_initializer()
        print("Initializing global variables...", self.mSess.run(init))
        
        # Setup summary writer with current graph
        if self.mLOG_GRAPH:
            self.mWriter = tf.summary.FileWriter(".\\Logs\\",self.mSess.graph)
        return
            
    def interact(self,state,step,metadata=False):
        assert step >= 0, "Error, must include step must be positive"

        if metadata == True:            
            print('Adding run metadata for step: ', step)
            run_options = tf.RunOptions(trace_level=tf.RunOptions.SOFTWARE_TRACE)
            run_metadata = tf.RunMetadata()
            [action,summaries,var_updates] = self.mSess.run(
                    self.mGraphOut,
                    feed_dict={self.mGraphIn: state},
                    options=run_options,
                    run_metadata=run_metadata)
            self.mWriter.add_run_metadata(run_metadata, 'S: %d' % i)
            self.mWriter.add_summary(summaries, global_step=step)
        else:  # Record a summary
            [action,summaries,var_updates] = self.mSess.run(self.mGraphOut, feed_dict={self.mGraphIn: state})
            self.mWriter.add_summary(summaries, global_step=step)        
        return [action,var_updates]

    def build_main_graph(self):            
        # Graph Inputs
        raw_data_new = tf.placeholder(tf.complex64,shape=[8,1000],name='p_raw_data_new')
    
        # Variables
        weights = tf.get_variable(name="v_weights",shape=[1,self.mFeatureSpaceSize],dtype=tf.float32,initializer=tf.constant_initializer(0))
        z_trace = tf.get_variable(name="v_ztrace",shape=[1,self.mFeatureSpaceSize],dtype=tf.float32,initializer=tf.constant_initializer(0))
        expected_reward = tf.get_variable(name="v_expected_reward",shape=[],dtype=tf.float32,initializer=tf.constant_initializer(0))
        self.mALPHA = tf.get_variable(name="alpha", shape=[],dtype=tf.float32,initializer=tf.constant_initializer( self._ALPHA))
                
        #STEP 1: Calculate actual reward
        #TODO actual_reward = self.mRewardExtract.map_reward(raw_data_new)
        actual_reward = self.mRewardExtract.map_reward_trivial(raw_data_new)
    
        #STEP 2: Find the error between expected and actual
        err_delta_1 = tf.subtract(actual_reward, expected_reward, name="o_calc_err_delta")
        
        #STEP 3: Determine if we're greedy or not    
        action_next = self.mActSel.map_next_action(raw_data_new,0,weights,self.mEPSILON)
        
        #STEP 5: Generate new S/A features
        bin_features =  tf.cast(self.mFeatExtract.activateBinaryFeatures_TF(action_next), dtype=tf.float32)
        
        #STEP 6: Generate new expected reward (used to update w,z,del)
        expected_reward_tmp = tf.reduce_sum(tf.multiply(tf.transpose(weights),bin_features))


        #STEP 7: Update error delta using look-ahead
        with tf.name_scope("ErrDel_Lookahead_Upd"):
            err_delta_2 = err_delta_1 + (self.mGAMMA * expected_reward_tmp)

        #STEP 8: Update our model

        with tf.name_scope("Weights_Upd"):
            weights_tmp = weights + (z_trace * err_delta_2 * self.mALPHA)
        
        with tf.control_dependencies([weights_tmp]):
            with tf.name_scope("Z_Upd"):
                z_trace_tmp = tf.transpose(bin_features) + (z_trace * self.mGAMMA * self.mLAMBDA) #decay, then update
                z_trace_tmp = tf.minimum(tf.ones([self.mFeatureSpaceSize]), z_trace_tmp) #replace all >1
            with tf.name_scope("AlphaDecay"):
                alpha_tmp = self.mALPHA * self.mALPHA_DECAY

        
        #UPDATES
        with tf.control_dependencies([err_delta_1]):
            with tf.name_scope("Updates"):
                weights_update = weights.assign(weights_tmp)
                z_trace_update =  z_trace.assign(z_trace_tmp)
                alpha_update = self.mALPHA.assign(alpha_tmp)
                expected_reward_update = expected_reward.assign(expected_reward_tmp)
        
        tf.summary.histogram("Weights",weights)
        tf.summary.histogram("EligibilityTrace",z_trace)
        tf.summary.histogram("BinFeat",bin_features)
        tf.summary.scalar("ExpectedReward",expected_reward)
        tf.summary.scalar("ActualReward",actual_reward)
        tf.summary.scalar("ErrorDelta",err_delta_2)
        tf.summary.histogram("ActionNextA",action_next)

        summaries = tf.summary.merge_all()
        
        var_updates_dict = {
                "Weights":weights,
                "EligibilityTrace":z_trace,
                "EligibilityTraceTmp":z_trace_tmp,
                "BinFeat":bin_features,
                "ExpectedReward":expected_reward,
                "ActualReward":actual_reward,
                "NextExpectedReward":expected_reward_tmp,
                "ErrorDelta":err_delta_2,
                "ActionNext":action_next,
                "Updates:":{
                        "Weights":weights_update,
                        "EligibilityTrace":z_trace_update,
                        "ExpectedReward":expected_reward_update,
                        "Alpha":alpha_update
                        }
                }
        
        var_updates = [weights_update,z_trace_update,expected_reward_update,self.mALPHA,alpha_update]
        
        return raw_data_new,[action_next,summaries,var_updates_dict]
        #return raw_data_new, action_next
    
    def generate_summaries(self):#,scalars,histograms):

        #STEP: Write summaries, return
        tf.summary.histogram("Weights",self.summaries[0])
        tf.summary.histogram("EligibilityTrace",self.summaries[1])
        tf.summary.histogram("ZTMP",self.summaries[2])
        tf.summary.histogram("BinFeat",self.summaries[3])
        tf.summary.scalar("ExpectedReward",self.summaries[4])
        tf.summary.scalar("ActualReward",self.summaries[5])
        tf.summary.scalar("ErrorDelta",self.summaries[6])
        tf.summary.histogram("ActionNextA",self.summaries[7])
        self.mAllSummaries = tf.summary.merge_all()
        
tf.reset_default_graph()
Sin10=[np.sin(2*np.pi * (x/250) * 10) for x in range(1000)]
state=[Sin10]

BBF = np.linspace(0, 50, 200)

GD=GradDescSarsaAgent()
RLE = RLEnv.RLEnv()

y=list()
z=list()
action=[0,0]
state=None
for i in range(10000):
    try:
        state = RLE.interact(action)
        #print(action[0], state[0][0])
        if i % 999 == 0:
            [a,b] = GD.interact(state,i,metadata=True)
        else:
            [a,b] = GD.interact(state,i)    
        #action = [BBF[a[0]],BBF[a[0]]]
        action = [a[0],a[0]]
        y.append(a)
        z.append(b)
#        z = b
    except KeyboardInterrupt:
        break