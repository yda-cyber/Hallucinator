#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 06:28:51 2022

@author: Dai-Bei Yang

Code adopted from:
    https://stackoverflow.com/a/28687555
"""

import numpy as np


def compute_euclidean_distances_matrix(x, y):
    x_square = np.expand_dims(np.einsum('ij,ij->i', x, x), axis=1)
    if x is y:
        y_square = x_square.T
    else:
        y_square = np.expand_dims(np.einsum('ij,ij->i', y, y), axis=0)
    distances = np.dot(x, y.T)
    # use inplace operation to accelerate
    distances *= -2
    distances += x_square
    distances += y_square
    # result maybe less than 0 due to floating point rounding errors.
    np.maximum(distances, 0, distances)
    if x is y:
        # Ensure that distances between vectors and themselves are set to large number.
        # This may not be the case due to floating point rounding errors.
        distances.flat[::distances.shape[0] + 1] = 1000.0
    return distances



def compute_contact_density(pos, cutoff=8):
    
    pos = pos[pos[:,2]=='CA'] 
    pos = pos[:,6:9].astype('float')
    length = len(pos)
    ca_dist_mat = compute_euclidean_distances_matrix(pos, pos)
    
    n_contact = ca_dist_mat<=cutoff**2
    n_contact_max = np.tri(length, k=-6).T
    
    n_contact = n_contact * n_contact_max
    # average_contact after 6 residues
    
    density = np.sum(n_contact)/np.sum(n_contact_max)

    return density


class ContactDensityLoss():
    
    def __init__(self, cutoff=8, target_density=0.035, factor=None,
                 max_loss=1, plddt_activate_value=0):
        
        # 100mer, 0.035; 200mer, 0.0155
        
        if factor is None:
            factor = -1/target_density
        self.factor = factor
        self.cutoff = cutoff
        self.target_density = target_density
        self.max_loss = max_loss
        self.plddt_activate_value = plddt_activate_value
    
    def calculate_loss(self, pos, plddt, job_name):
    
        if np.mean(plddt) < self.plddt_activate_value:
            return self.max_loss, {}
        
        density = compute_contact_density(pos, self.cutoff)
        loss = self.factor * (density-self.target_density) * (density < self.target_density)
        loss = self.max_loss * min(1,max(loss, 0))
        
        return loss, {'Contact %': np.round(100*density,2)}
    
    def callback(self, pos, job_name):
        
        pass