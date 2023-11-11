#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 22:31:37 2022

@author: exouser
"""

import warnings
import numpy as np
import pandas as pd
import pyKVFinder as kv
#from itertools import product
#from scipy.spatial.transform import Rotation as R
from matplotlib import pyplot as plt

from Hallucinator.loss.Compute_Cavity_Volume_Loss import reform_kvformat_pos
from Hallucinator.loss.Compute_Contact_Density_Loss import compute_euclidean_distances_matrix
from Hallucinator.loss.Compute_Cavity_Volume_Loss import CavityVolumeLoss


plt.rcParams['figure.dpi'] = 600

warnings.filterwarnings("ignore")
# Warning is ignored (nan)

shell_sectors = np.array([
    [0.356822,   0.,        0.934172],
    [-0.57735,  -0.57735,   -0.57735],
    [-0.57735,  -0.57735,    0.57735],
    [-0.57735,   0.57735,   -0.57735],
    [-0.57735,   0.57735,    0.57735],
    [0.57735,   -0.57735,   -0.57735],
    [0.57735,   -0.57735,    0.57735],
    [0.57735,    0.57735,   -0.57735],
    [0.57735,    0.57735,    0.57735],
    [0.934172,   0.356822,        0.],
    [0.934172,  -0.356822,        0.],
    [-0.934172,  0.356822,        0.],
    [-0.934172, -0.356822,        0.],
    [-0.356822,  0.,       -0.934172],
    [0.356822,   0.,       -0.934172],
    [0.,         0.934172,  0.356822],
    [0.,         0.934172, -0.356822],
    [0.,        -0.934172,  0.356822],
    [0.,        -0.934172, -0.356822],
    [0.,         0.525731,  0.850651],
    [0.,        -0.525731,  0.850651],
    [0.850651,   0.,        0.525731],
    [0.525731,  -0.850651,        0.],
    [0.525731,   0.850651,        0.],
    [0.,        -0.525731, -0.850651],
    [-0.850651,  0.,       -0.525731],
    [-0.525731, -0.850651,        0.],
    [-0.850651,  0.,        0.525731],
    [0.850651,   0.,       -0.525731],
    [0.,         0.525731, -0.850651],
    [-0.525731,  0.850651,        0.]])

shell_sectors = shell_sectors * 1 / np.linalg.norm(shell_sectors,
                                                   axis=1).reshape(-1, 1)
# Normalized

def plot_surf(surf, surf2=None):
    
    alpha=0.3 if surf2 is None else 0.1
    surf = np.copy(surf) 
    surf -= np.mean(surf, axis=0)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(surf[:,0], surf[:,1], surf[:,2], alpha=alpha)
    if surf2 is not None:
        surf2 = np.copy(surf2) 
        surf2 -= np.mean(surf2, axis=0)
        ax.scatter3D(surf2[:,0], surf2[:,1], surf2[:,2])


def point_to_line_distance(point):
    # d(A,BC) = magnitude(cross(A - B, C - B)) / magnitude(C - B)
    # In this case, A is the point on the surface, B is original, C is sector mean
    dist = np.linalg.norm(np.cross(point, shell_sectors), axis=1)
    #print(dist)
    idx = np.argmin(dist)
    return dist[idx], idx

    # I should not use this. Need to have directions
    
def point_to_line_distance_with_direction(point):
    
    dist = np.linalg.norm(np.cross(point, shell_sectors), axis=1)  
    idx = np.argsort(dist)[:2]
    s0,s1 = shell_sectors[idx[0]], shell_sectors[idx[1]]
    d1,d2 = np.linalg.norm(point-s0), np.linalg.norm(point-s1)
    return (dist[idx[0]], idx[0]) if d1<d2 else (dist[idx[1]], idx[1])


def compute_distribution(surf, bins, sort_order=None, plot=False, 
                         normalized=True, center=None):

    surf = np.copy(surf)
    if center is None:
        mid_point = np.mean(surf, axis=0)
    else:
        mid_point = np.asarray(center)

    dist_to_center = compute_euclidean_distances_matrix(surf,
                                                        mid_point.reshape(1, -1))
    dist_to_center = np.sqrt(dist_to_center)

    if type(bins) is int:
        bins = np.concatenate((np.linspace(0, np.max(dist_to_center)+0.01, bins),
                               np.array([np.inf])))
        
    surf -= mid_point

    sector_idx = np.asarray([*map(lambda x: point_to_line_distance_with_direction(x)[1], surf)])
    unq, counts = np.unique(sector_idx, return_counts=1)

    unsort_sector_hist = np.zeros(len(shell_sectors))
    unsort_sector_hist[unq] = counts

    if sort_order is None:
    #if True:
        sort_order = np.argsort(unsort_sector_hist)

    # Is this needed?
    #sorted_sector_hist = unsort_sector_hist[sort_order]
    #sorted_sector_hist /= len(surf)

    so_distr = np.zeros((len(shell_sectors), len(bins)-1))
    for i, sector in enumerate(sort_order):
        surf_sector = dist_to_center[sector_idx == sector]
        if len(surf_sector) == 0:
            continue
        sec_distr, _ = np.histogram(surf_sector, bins=bins)
        so_distr[i] = sec_distr

    if normalized: so_distr /= len(surf)
    if plot: plt.imshow(so_distr)

    return so_distr, bins, sort_order


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

    surf, volu, area = kv.spatial(cavs, step=step)
    surf = convert_kvcav_xyz(surf, ncav+2, vet, step)

    so_distr, bins, so = compute_distribution(surf, sample_points)

    return so_distr, bins, so, list(volu.items())[ncav][1]


def convert_kvcav_xyz(cavs, cav_index, vet, step):

    o = vet[0]
    i, j, k = vet[1]-o, vet[2]-o, vet[3]-o
    i, j, k = i/np.linalg.norm(i), j/np.linalg.norm(j), k/np.linalg.norm(k)
    coord = o + (np.argwhere(cavs == cav_index)*step)@np.array([i, j, k])

    return np.asarray(coord).astype('float')


class CavitySimilarityLoss():

    def __init__(self, pdb_guest, vdw_guest, volume_factor, volume_expansion,
                 similarity_factor, similarity_target_diff, sample_points=10,
                 step_guest=0.2, step_protein=0.6, backbone_cavity=True,
                 max_loss=5, plddt_activate_value =50):

        pos = pd.read_csv(pdb_guest, sep='\s+', header=None)
        self.so_distr, self.bins, self.order, self.target_volume = compute_initial_distribution(
            pos, vdw_guest, sample_points, step_guest)

        print('[CSLOG]: Targeted volume is %.2f' % self.target_volume)

        self.target_diff = similarity_target_diff
        self.factor = similarity_factor
        self.step_protein = step_protein
        self.backbone_cavity = backbone_cavity
        self.volume_loss = CavityVolumeLoss(step_protein, self.target_volume,
                                            volume_factor, volume_expansion,
                                            1, plddt_activate_value)
        self.plddt_activate_value = plddt_activate_value
        self.max_loss = max_loss

    def calculate_loss(self, pos, plddt, job_name, output_filename=None):
        
        if np.mean(plddt) < self.plddt_activate_value:
            return self.max_loss, {}

        new_pos = reform_kvformat_pos(pos, self.backbone_cavity)
        vet = kv.get_vertices(new_pos, step=self.step_protein)
        ncav, cavs = kv.detect(new_pos, vet, step=self.step_protein)
        if ncav == 0:
            loss_vol, info_vol = self.volume_loss.calculate_loss(pos, [100,100],
                                                                 job_name,
                                                                 vol=np.array([0]))
            loss_sim, info_sim = 1, np.inf
        else:
            surf, volu, _ = kv.spatial(cavs, step=self.step_protein)
            volu = np.asarray(list(volu.values()))
            volu_diff = volu - self.target_volume
            volu_diff[volu_diff < 0] = volu_diff[volu_diff < 0] * 3  # Penalty
            idx = np.argmin(abs(volu_diff))
            # Find the cavity with smallest difference of volume

            loss_vol, info_vol = self.volume_loss.calculate_loss(pos, [100,100],
                                                                 job_name, 
                                                                 np.array([volu[idx]]))
            surf = convert_kvcav_xyz(surf, idx+2, vet, self.step_protein)

            so_distr_protein, _, _ = compute_distribution(
                surf, self.bins, None)

            # Use a distance-like object desrcibe differences
            diff = np.sqrt(np.sum((self.so_distr - so_distr_protein)**2))
            info_sim = diff
            # Sigmoid at x=-5 -> 0
            loss_sim = 1/(1+np.exp(-(-5+self.factor*(diff-self.target_diff))))
            if np.isnan(loss_sim) or np.isinf(loss_sim):
                loss_sim = 1

        if output_filename is not None:

            kv.export(output_filename, cavs, None, vet, step=self.step_protein)

        info_vol['Similar'] = np.round(info_sim, 2)
        
        return self.max_loss*(loss_sim+loss_vol-loss_sim*loss_vol), info_vol
    
    def callback(self, pos, job_name):
        
        self.calculate_loss(pos, 100, job_name,
                            './results/'+job_name+'/cavity_similar.pdb')
