#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 22:03:51 2022

@author: Dai-Bei Yang
"""

import numpy as np

def select_mutation(length, num_mutation, plddt=None, factor=0.05, random=np.random):
    
    if plddt is None:
        mutations = random.choice(range(length), size=num_mutation)
    else:
        partition = np.exp(-1*factor*plddt)
        prob = partition/np.sum(partition)
        mutations = random.choice(range(length), size=num_mutation, p=prob,
                                     replace=False)
        
    return mutations