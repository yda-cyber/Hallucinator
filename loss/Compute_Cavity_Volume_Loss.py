#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 05:04:50 2022

@author: exouser
"""

import pyKVFinder as kv
import numpy as np


def reform_kvformat_pos(pos, backbone=False, vdw=None):

    if backbone:
        pos = pos[(pos[:, 2] == 'N') | (pos[:, 2] == 'C') |
                  (pos[:, 2] == 'CA') | (pos[:, 2] == 'O')]
    #pos = pos[(pos[:,2]=='CA')]
    if vdw is None:
        vdw = kv.read_vdw()
    new_pos = [*map(lambda p: [p[5], p[4], p[3], p[2], p[6], p[7], p[8],
                               vdw[p[3]][p[2]]], pos)]
    return np.asarray(new_pos)


def compute_cavity_volume(pos, step, output_filename):

    pos = reform_kvformat_pos(pos)
    vet = kv.get_vertices(pos, step=step)
    ncav, cavs = kv.detect(pos, vet, step=step)
    
    if output_filename is not None:
        kv.export(output_filename, cavs, None, vet, step=step)
        
    if ncav != 0:
        _, volu, _ = kv.spatial(cavs, step=step)
        return np.asarray(list(volu.values()))
    else:
        return np.array([0])


class CavityVolumeLoss():

    def __init__(self, cavity_step, target_volume, factor, expansion, max_loss,
                 plddt_activate_value):
        
        # The cavity_step will strongly influence the speed.
        self.cavity_step = cavity_step
        self.target_volume = target_volume
        self.factor = factor
        self.expansion = expansion
        self.max_loss = max_loss
        self.plddt_activate_value = plddt_activate_value

    def calculate_loss(self, pos, plddt, job_name=None, vol=None):
        
        if np.mean(plddt) < self.plddt_activate_value:
            return self.max_loss, {}
        
        if vol is None:
            vol = compute_cavity_volume(pos, self.cavity_step)
        vol = vol.astype('float')-float(self.target_volume)

        dvol = np.min(abs(vol))

        sigm = 1/(1+np.exp(-self.factor*(dvol-self.expansion))) - \
            1/(1+np.exp(-self.factor*(dvol+self.expansion)))
        efac = 1/(1+np.exp(-self.factor*(-self.expansion))) - \
            1/(1+np.exp(-self.factor*(+self.expansion)))

        return (self.max_loss* (sigm-efac)/abs(efac), 
                               {'Diff volume': np.round(dvol, 2)})
    
    
    def callback(self, pos, job_name):
        
        vol = compute_cavity_volume(pos, self.cavity_step, 
                                    './results/'+job_name+'/cavity_volume.pdb')
