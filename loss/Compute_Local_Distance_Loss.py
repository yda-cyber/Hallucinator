#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 18:59:52 2023

@author: exouser
"""

import numpy as np
import pandas as pd


def compute_distance(x1, x2):
    x1, x2 = x1.astype('float'), x2.astype('float')
    return np.sqrt(np.sum((x1-x2)**2))


class LocalDistanceLoss():

    def __init__(self, resid, target_distance, Cb_close=True, k=10, dx=5,
                 max_loss=10, plddt_activate_value=30, long_axis_extend=False):

        self.resid = resid[0]-1, resid[1]-1
        self.target_distance = target_distance
        self.Cb_close = Cb_close
        self.max_loss = max_loss
        self.plddt_activate_value = plddt_activate_value
        self.long_axis_extend = long_axis_extend
        self.k = k
        self.dx = dx

    def calculate_loss(self, pos, plddt, job_name, dirs):

        if np.mean(plddt) < self.plddt_activate_value:
            return self.max_loss, {}

        ca = pos[pos[:, 2] == 'CA']
        ca1, ca2 = ca[self.resid[0]], ca[self.resid[1]]
        plddt_eff = 1/2*(plddt[self.resid[0]]+plddt[self.resid[1]])
        res1, res2, idx1, idx2, rid1, rid2 = ca1[3], ca2[3], ca1[1] - \
            1, ca2[1]-1, ca1[5], ca2[5]

        cb1, cb2 = None, None
        if res1 == 'GLY':
            cb1 = ca1
        if res2 == 'GLY':
            cb2 = ca2
        if cb1 is None:
            part_pos = pos[idx1:idx1+10]
            cb1 = part_pos[part_pos[:, 2] == 'CB'][0]
        if cb2 is None:
            part_pos = pos[idx2:idx2+10]
            cb2 = part_pos[part_pos[:, 2] == 'CB'][0]

        assert cb1[5] == rid1
        assert cb2[5] == rid2

        d1 = compute_distance(ca1[6:9], ca2[6:9])
        d2 = compute_distance(cb1[6:9], cb2[6:9])

        if self.Cb_close is None:
            sigm = 0
        elif (d2 <= d1) != self.Cb_close:
            sigm = 1
        else:
            sigm = 0
        
        if sigm == 0:
            if self.long_axis_extend:
                sigm = 1/(1+np.exp(-self.k/10*((d1-self.target_distance)-self.dx))
                          ) - 1/(1+np.exp(-self.k*((d1-self.target_distance)))) + 1
            else:
                sigm = 1/(1+np.exp(-self.k*((d1-self.target_distance)-self.dx/2))
                          ) - 1/(1+np.exp(-self.k*((d1-self.target_distance)+self.dx/2))) + 1

        sigm = min(1, max(sigm, 0))
        sigm = min(1, sigm / min(1, plddt_eff/70)**2)

        return self.max_loss * sigm, {'Dist': (np.round(d1, 1), np.round(plddt_eff, 2), (d2 <= d1) == self.Cb_close)}

    def callback(self, pos, job_name, dirs):

        pass
