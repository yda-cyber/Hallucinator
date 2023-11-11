#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 18:59:52 2023

@author: exouser
"""

import numpy as np
import pandas as pd
from itertools import combinations

from Hallucinator.loss.Compute_Local_Distance_Loss import LocalDistanceLoss, compute_distance



class PreserveStructureLoss():

    def __init__(self, preserve_resid, pos_file, k=10, max_loss=10, 
                 plddt_activate_value=0):

        self.resid = preserve_resid
        pos = pd.read_csv(pos_file, sep='\s+', header=None).to_numpy()
        ca = pos[pos[:, 2] == 'CA']
        ca = ca[:, 6:9].astype('float')
        self.resid_combs = list(combinations(self.resid,2))
        self.dist_combs = [*map(lambda x: compute_distance(ca[x[0]-1], ca[x[1]-1]), 
                                self.resid_combs)]
        self.loss_combs = [*map(lambda i: LocalDistanceLoss(self.resid_combs[i], 
                                                            self.dist_combs[i], Cb_close=None, 
                                                            k=k, dx=1/3*self.dist_combs[i], 
                                                            max_loss=max_loss/len(self.dist_combs)),
                                range(len(self.dist_combs)))]
        
        self.max_loss = max_loss
        self.plddt_activate_value = plddt_activate_value

    def calculate_loss(self, pos, plddt, job_name, print_detail=False):

        if np.mean(plddt) < self.plddt_activate_value:
            return self.max_loss, {}

        loss = [*map(lambda x: x.calculate_loss(pos, plddt, job_name)[0],
                     self.loss_combs)]
        if print_detail:
            ca = pos[pos[:,2]=='CA'][:, 6:9].astype('float')
            dist = [*map(lambda x: compute_distance(ca[x[0]-1], ca[x[1]-1]), 
                                    self.resid_combs)]
            for i in range(len(dist)):
                print(dist[i], self.dist_combs[i], loss[i])

        loss = np.sum(loss)
        return loss, {'Preserve': np.round(loss/self.max_loss, 2)}

    def callback(self, pos, job_name):

        pass
