#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 00:20:55 2023

@author: exouser
"""

import numpy as np
from Hallucinator.modules.Compute_Secondary_Structure_Ratio import compute_secondary_structure_ratio

def vshape_function(x, target, ext):

    y = -1/(target-ext) * (x-target+ext) * (x < target-ext) + 0 + \
            1/(1-target-ext) * (x-target-ext) * (x > target+ext)
            
    return y 


class SecondaryRatioLoss():

    def __init__(self, ratio, extent, length_cutoff=(6, 4),
                 plddt_activate_value=0, max_loss=5):

        self.helix, self.sheet = ratio
        # need to be smaller than 1

        self.helix_ext, self.sheet_ext = extent
        #self.helix_par, self.sheet_par = parameter
        self.helix_cut, self.sheet_cut = length_cutoff
        self.plddt_activate_value = plddt_activate_value
        self.max_loss = max_loss


    def calculate_loss(self, pos, plddt, job_name):

        if np.mean(plddt) < self.plddt_activate_value:
            return self.max_loss, {}

        ratio = compute_secondary_structure_ratio(pos, helx_cutoff=self.helix_cut,
                                                  anti_cutoff=self.sheet_cut,
                                                  parr_cutoff=self.sheet_cut)

        helix, sheet = ratio[0], ratio[1]+ratio[2]
        helix_ext, sheet_ext = self.helix_ext/200, self.sheet_ext/200

        loss_helix = vshape_function(helix, self.helix/100, helix_ext)
        loss_sheet = vshape_function(sheet, self.sheet/100, sheet_ext)
        loss_helix = 1 if loss_helix >= 1 else loss_helix
        loss_sheet = 1 if loss_sheet >= 1 else loss_sheet

        loss = loss_helix/2 + loss_sheet/2

        return self.max_loss*loss, {'Helix %': np.round(helix*100, 0), 'Sheet %': np.round(sheet*100, 0)}

    def callback(self, pos, job_name):

        pass
