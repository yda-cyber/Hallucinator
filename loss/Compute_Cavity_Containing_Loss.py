#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 04:27:54 2023

@author: exouser
"""

import numpy as np
import pandas as pd
import pyKVFinder as kv

from loss.Compute_Cavity_Similarity_Loss import plot_surf
from loss.Compute_Cavity_Volume_Loss import CavityVolumeLoss
from loss.Compute_Cavity_Volume_Loss import reform_kvformat_pos
from loss.Compute_Cavity_Similarity_Loss import convert_kvcav_xyz
from loss.Compute_Cavity_Similarity_Loss import compute_distribution
from loss.Compute_Contact_Density_Loss import compute_euclidean_distances_matrix


# Use confab to create multi-pose

def compute_initial_distribution(pos_guest, vdw_guest, sample_points=10, step=0.3):

    vdw = kv.read_vdw()
    for k, v in vdw_guest.items():
        vdw[k] = v
    # Add guest vdw

    new_pos = reform_kvformat_pos(
        np.asarray(pos_guest), backbone=False, vdw=vdw)
    vet = kv.get_vertices(new_pos, step=step)
    ncav, cavs = kv.detect(new_pos, vet, step=step)
    cavs[cavs == 0] = ncav+2
    cavs[cavs == -1] = 0

    vol_cav = convert_kvcav_xyz(cavs, ncav+2, vet, step)
    surf, volu, area = kv.spatial(cavs, step=step)
    so_distr, bins, so = compute_distribution(vol_cav, sample_points,
                                              normalized=False)

    return so_distr, bins, so, list(volu.items())[ncav][1], vol_cav


class CavityContainingLoss():

    def __init__(self, pdb_guest, vdw_guest, volume_factor, volume_expansion,
                 similarity_factor=40, similarity_target_diff=0, 
                 sample_points=10, step=0.3, backbone_cavity=False,
                 max_loss=5, plddt_activate_value=50, use_effective_score=True):

        pos = pd.read_csv(pdb_guest, sep='\s+', header=None)
        self.so_distr, self.bins, self.order, self.target_volume, self.vol_cav = compute_initial_distribution(
            pos, vdw_guest, sample_points, step)

        print('[CSLOG]: Targeted volume is %.2f' % self.target_volume)

        self.target_diff = similarity_target_diff
        self.factor = similarity_factor
        self.step = step
        self.npoints = np.sum(self.so_distr)
        self.backbone_cavity=backbone_cavity
        self.volume_loss = CavityVolumeLoss(self.step, self.target_volume,
                                            volume_factor, volume_expansion,
                                            1, plddt_activate_value)
        self.max_loss = max_loss
        self.plddt_activate_value = plddt_activate_value
        self.use_effective_score = use_effective_score
        
    def calculate_loss(self, pos, plddt, job_name, output_filename = None):
        
        if np.mean(plddt) < self.plddt_activate_value:
            return self.max_loss, {}

        new_pos = reform_kvformat_pos(pos, self.backbone_cavity)
        vet = kv.get_vertices(new_pos, step=self.step)
        ncav, cavs = kv.detect(new_pos, vet, step=self.step)
        if ncav == 0:
            loss_vol, info_vol = self.volume_loss.calculate_loss(pos, [100,100],
                                                                 job_name,
                                                                 np.array([0]))
            loss_ctn, info_ctn = 1, np.inf
        else:
            surf, volu, _ = kv.spatial(cavs, step=self.step)
            volu = np.asarray(list(volu.values()))
            volu_diff = volu - self.target_volume
            volu_diff[volu_diff < 0] = np.inf
            idx = np.argmin(abs(volu_diff))

            vol_cav = convert_kvcav_xyz(cavs, idx+2, vet, self.step)
            so_distr_protein, _, _ = compute_distribution(
                vol_cav, self.bins, None, normalized=False)

            diff_distr = so_distr_protein-self.so_distr
            v1 = -1*np.sum(diff_distr[diff_distr < 0])/self.npoints
            
            info_ctn = v1
            loss_ctn = 1/(1+np.exp(-(-5+self.factor*(v1-self.target_diff))))
            
            loss_vol, info_vol = self.volume_loss.calculate_loss(pos, [100,100],
                                                                 job_name,
                                                                 np.array([volu[idx]]))
            loss_vol = 1 if v1 > 0.4 else loss_vol
            
        if output_filename is not None:
            kv.export(output_filename, cavs, None, vet, step=self.step)
            plot_surf(vol_cav, self.vol_cav)
        
        if self.use_effective_score and ncav !=0 :
            residues = kv.constitutional(cavs, new_pos, vet, step=self.step)
            ef_resid = np.asarray(residues[list(residues)[idx]])[:,0].astype('int')
            ef_plddt = plddt[ef_resid]
            if np.mean(ef_plddt) < self.plddt_activate_value:
                loss_ctn, info_ctn = 1, 'Unstable'
            #elif np.any(ef_plddt < self.plddt_activate_value - 20):
            #    loss_ctn, info_ctn = 1, 'Unstable'              
            else:
                info_ctn = np.round(info_ctn, 2)
            loss_ctn = min(1, loss_ctn / min(1,np.min(ef_plddt)/70)**2)

        info_vol['ConFactor'] = (info_ctn,np.round(loss_ctn,2),np.round(loss_vol,2))
        loss_ctn = self.max_loss*(loss_ctn+loss_vol)/2
        
        return loss_ctn, info_vol
    
    
    def callback(self, pos, job_name):
        
        flag = self.use_effective_score
        self.use_effective_score = 0
        self.calculate_loss(pos, np.array([100,100])
                            , job_name,  './results/'+job_name+'/cavity_contain.pdb')
        self.use_effective_score = flag