# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 03:46:36 2017

@author: MartianMartin
"""

'''
For realtime beta power (assuming known target frequency) we need the following steps.

1) Check last 1 second epoch for AMP > 150e-6 V, throw out if greater

2) Assuming desired fps=60, we need to compute new point every ~4 samples
   ==> dot product buffer with wavelet of len ~256 

3) Compute magnitude i.e. np.abs(inner_product_from_step_2)

THATS IT! :)

CAVEATS:

    - will need to interpolate signal after throwing out noisy point(s)
    - TBD
    
    
    
'''