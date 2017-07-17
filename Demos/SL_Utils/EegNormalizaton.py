# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 17:57:57 2017

@author: marzipan
"""

'''
- Baseline period normalization (e.g. log-norm) may amplify intra-trial effects
  while eliminating inter-trial differences.
  
- Proposal:
    
    - Entrainment effects (task-related) (intra-trial) 
    
        - Can be scaled using decibel conversion,
          i.e. dB_tf = 10*log10(activity_tf / baseline_f)
        - baseline typicall ~300 ms
        - symmetric color scaling preffered
        
        - Others normalization transforms
            - percentage change
            - Ztf
        
        - Important parameters
        
            - Choise of baseline window
                - separate rest period ~10 s
                - pre-trial period ~300 ms
                - entire trial as baseline
                - control condition as baseline
                - condition-specific or condition-average baseline
                
            - Number of trials
                - 20 for error-related frontal theta (low-end...)
                - Low frequencies require fewer trials
                - Compare time course of 1 trial to average for robustness estimate
                
            - Downsample after time-frequency decomposition
                
      
    - Gross state comparision (task-unrelated) (inter-trial) 
    
        - SNR ~ mean/std
        - 
        
NOTES:
    
    1) We need to choose something specific to look for to focus analysis.
    2)  

Misc:
    
    The general approach in eeg data analysis is to expose subject to various
    conditions, and then try to find differences in the data across these 
    conditions. Because differences are not obvious, the data must be transformed,
    with most of the complexity of research being contained in choosing the right
    transform to reveal the difference. 
    
    In principle, we could neural networks to determine the presence of differences
    for us, where the different conditions determine the labels. 
    
    Note that in describing how two conditions are different, features can be
    extract along many dimensions including time, frequency, and space. 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
'''