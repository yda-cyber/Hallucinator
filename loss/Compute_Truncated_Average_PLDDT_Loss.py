#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 20:10:22 2023

@author: exouser
"""

import numpy as np


class TruncatedAveragePLDDTLoss():
    
    def __init__(self, min_val, max_val):
        
        self.min_val = min_val
        self.max_val = max_val
        self.max_loss = 1
    
    def calculate_loss(self, plddt, job_name):
        
        m = np.mean(plddt)
        return (min(1,max(0,(1-(m-self.min_val)/(self.max_val-self.min_val)))), 
                {'Ave Plddt': np.round(m, 2)})
    
    def callback(self, plddt, job_name):
        
        pass
