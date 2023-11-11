#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 22:03:51 2022

@author: Dai-Bei Yang
"""

import numpy as np

def select_mutation(length, num_mutation, plddt=None, factor=0.05, 
                    random=np.random, preserve_resid=None):
    
    if plddt is None:
        mutations = random.choice(range(length), size=num_mutation, replace=False)
    else:
        partition = np.exp(-1*factor*plddt)
        if preserve_resid is not None:
            partition[preserve_resid-1] = 0
        prob = partition/np.sum(partition)
        mutations = random.choice(range(length), size=num_mutation, p=prob,
                                     replace=False)
        
    return mutations