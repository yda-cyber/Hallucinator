#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 04:27:54 2023

@author: exouser
"""

import numpy as np
import pandas as pd
import pyKVFinder as kv
from scipy.optimize import minimize

from Hallucinator.loss.Compute_Cavity_Volume_Loss import reform_kvformat_pos
from Hallucinator.loss.Compute_Cavity_Similarity_Loss import convert_kvcav_xyz
from Hallucinator.loss.Compute_Cavity_Similarity_Loss import compute_distribution
from Hallucinator.loss.Compute_Contact_Density_Loss import compute_euclidean_distances_matrix


# Use confab to create multi-pose

def compute_initial_distribution(pos_host, vdw_host, sample_points=10, step=0.3,
                                 backbone_cavity=True):

    vdw = kv.read_vdw()
    if vdw_host is not None:
        for k, v in vdw_host.items():
            vdw[k] = v
    # Add guest vdw

    new_pos = reform_kvformat_pos(
        np.asarray(pos_host), backbone=backbone_cavity, vdw=vdw)
    vet = kv.get_vertices(new_pos, step=step)
    ncav, cavs = kv.detect(new_pos, vet, step=step)
    _, volu, _ = kv.spatial(cavs, step=step)
    idx = np.argmax(np.asarray(list(volu.values())))

    vol_cav = convert_kvcav_xyz(cavs, idx+2, vet, step)
    surf, volu, area = kv.spatial(cavs, step=step)
    so_distr, bins, so = compute_distribution(vol_cav, sample_points,
                                              normalized=False)

    return so_distr, bins, so, list(volu.items())[idx][1], vol_cav



class ProteinContainingLoss():

    def __init__(self, pdb_host=None, shape_function=None, max_points=100, similarity_factor=10, similarity_target_diff=0, 
                 sample_points=20, step=0.5, backbone_cavity=True, max_loss=5,
                 plddt_activate_value=50, dirs):

        if pdb_host is not None:
            pos = pd.read_csv(pdb_host, sep='\s+', header=None)
            self.so_distr, self.bins, self.order, _, self.vol_cav = compute_initial_distribution(
                pos, None, sample_points, step)
            
        if shape_function is not None:
            grid1D = np.linspace(-max_points, max_points, 2*max_points+1) * step
            x,y,z = np.meshgrid(grid1D,grid1D,grid1D)
            grid3D = np.c_[x.ravel(), y.ravel(), z.ravel()]
            grid_acc = grid3D[np.asarray([*map(shape_function, grid3D)])]
            self.so_distr, self.bins, self.order = compute_distribution(grid_acc, sample_points,
                                                      normalized=False)
            self.vol_cav = grid_acc
        print('[PSLOG]: Targeted host volume generated.')

        self.target_diff = similarity_target_diff
        self.factor = similarity_factor
        self.step = step
        self.npoints = np.sum(self.so_distr)
        self.backbone_cavity=backbone_cavity

        self.max_loss = max_loss
        self.plddt_activate_value = plddt_activate_value


    def calculate_loss(self, pos, plddt, job_name, dirs):
        
        if np.mean(plddt) < self.plddt_activate_value:
            return self.max_loss, {}
        
        new_pos = reform_kvformat_pos(
            np.asarray(pos), backbone=self.backbone_cavity)
        vet = kv.get_vertices(new_pos, step=self.step)
        ncav, cavs = kv.detect(new_pos, vet, step=self.step)
        vet = kv.get_vertices(new_pos, step=self.step)
        ncav, cavs = kv.detect(new_pos, vet, step=self.step)
        cavs[cavs == 0] = ncav+2
        cavs[cavs == -1] = 0
        
        vol_cav = convert_kvcav_xyz(cavs, ncav+2, vet, self.step)  
        _, volu, _ = kv.spatial(cavs, step=self.step)
        
        so_distr_protein, bins, so = compute_distribution(vol_cav, self.bins,
                                                  normalized=False)
        
        diff_distr = -so_distr_protein+self.so_distr
        v1 = -1*np.sum(diff_distr[diff_distr < 0])/np.sum(so_distr_protein)
            
        info_ctn = {'Guest': np.round(v1,2)}
        loss_ctn = 1/(1+np.exp(-(-5+self.factor*(v1-self.target_diff))))
        loss_ctn = min(1, loss_ctn / min(1,np.mean(plddt)/70)**2)
        
        loss_ctn = self.max_loss*(loss_ctn)
        
        return loss_ctn, info_ctn
    
    def callback(self, pos, job_name, dirs):
        
        pass
