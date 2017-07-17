# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 12:08:12 2017

@author: MartianMartin
"""

import numpy as np

class RLEnv:
    
    def __init__(self):
        self.TARGET_FREQUENCY_HZ = 10 
        self.TARGET_FREQUENCY__2_HZ = 25
        self.MAX_FREQUENCY_DIFF_HZ = 10
        self.MAX_FREQUENCY_DIFF_2_HZ = 20
        self.SRATE = 250
        self.WIN_LEN = 1000
        self.BASE_STATE = np.sin([2*np.pi*self.TARGET_FREQUENCY_HZ*n/self.SRATE for n in range(self.WIN_LEN)])
    
    def interact(self, action):
        '''
        input: new action [float(amp), float(freq)] that agent submits to environment.
        output: new state that environment generates in response to latest action.
        '''
        return self.mode_2(action)

    def mode_1(self, action):
        '''
        Returns *no change* unless 10 Hz BB is received for SUSTAIN_CRITERIA
        consecutive time steps
        '''
        pass
    
    def mode_2(self, action):
        '''
        Returns less reward if BB frequency is further from 10  Hz, more if closer.
        '''
        freq = action[0]
        diff = abs(self.TARGET_FREQUENCY_HZ - freq)
        inv_reward_magnitude = diff if (diff < self.MAX_FREQUENCY_DIFF_HZ) else self.MAX_FREQUENCY_DIFF_HZ
        reward_magnitude = self.MAX_FREQUENCY_DIFF_HZ - inv_reward_magnitude
        new_state = reward_magnitude * self.BASE_STATE
        return [new_state]*8
    
    def mode_3(self, action):
        '''
        Returns *no change* unless specific melody sequence is followed
        '''
        pass
    
    def mode_4(self, action):
        '''
        Two maxima, one higher than the other
        '''
        freq = action[0]
        
        if freq > 20:
            
            diff = abs(self.TARGET_FREQUENCY__2_HZ - freq)
            inv_reward_magnitude = diff if (diff < self.MAX_FREQUENCY_DIFF_2_HZ) else self.MAX_FREQUENCY_DIFF_2_HZ
            reward_magnitude = self.MAX_FREQUENCY_DIFF_2_HZ - inv_reward_magnitude
            new_state = reward_magnitude * self.BASE_STATE
        
        else:
        
            diff = abs(self.TARGET_FREQUENCY_HZ - freq)
            inv_reward_magnitude = diff if (diff < self.MAX_FREQUENCY_DIFF_HZ) else self.MAX_FREQUENCY_DIFF_HZ
            reward_magnitude = self.MAX_FREQUENCY_DIFF_HZ - inv_reward_magnitude
            new_state = reward_magnitude * self.BASE_STATE
            
        return [new_state]*8
        #return reward_magnitude
    
    
    
    
    
    
    
    
    
    
    