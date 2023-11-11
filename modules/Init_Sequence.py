#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 05:23:27 2022

@author: Dai-Bei Yang

Code adopted from : 
    https://github.com/bwicky/oligomer_hallucination/blob/main/oligomer_hallucination.py

"""

import numpy as np


def init_sequence(length, excluded_aas=[], custom_init_freq=None, random=np.random):
    
    if custom_init_freq is None:
        AA_freq = np.copy(BACKGROUND)
    else:
        assert len(custom_init_freq) == 20, 'Must Have Size 20!'
        AA_freq = np.assary(custom_init_freq)
        
    for aa in excluded_aas:
        ind = np.argwhere(AAORDER==aa)[0][0]
        AA_freq[ind] = 0
    
    AA_freq = AA_freq / AA_freq.sum()
    AA_freq= {a:f for a, f in list(zip(AAORDER, AA_freq))}
    init_sequence = ''.join(random.choice(list(AA_freq.keys()), size=length, 
                                              p=list(AA_freq.values())))
    
    return init_sequence
    

# BLOUSM62
BACKGROUND = [7.4216205067993410e-02, 5.1614486141284638e-02, 4.4645808512757915e-02,
 5.3626000838554413e-02, 2.4687457167944848e-02, 3.4259650591416023e-02,
 5.4311925684587502e-02, 7.4146941452644999e-02, 2.6212984805266227e-02,
 6.7917367618953756e-02, 9.8907868497150955e-02, 5.8155682303079680e-02,
 2.4990197579643110e-02, 4.7418459742284751e-02, 3.8538003320306206e-02,
 5.7229029476494421e-02, 5.0891364550287033e-02, 1.3029956129972148e-02,
 3.2281512313758580e-02, 7.2919098205619245e-02];

AAORDER = np.array(list('ARNDCQEGHILKMFPSTWYV'))