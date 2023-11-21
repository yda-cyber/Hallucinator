#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 06:44:19 2022

@author: Dai-Bei Yang
"""

import numpy as np


class LossFunction():

    def __init__(self, plddt_func=[], pos_func=[],form = lambda x,y: x+y, 
                 info_print=False, pos_func_rule=np.max, plddt_func_rule=np.max,
                 logger=None, save_dirs='./'):
        
        self.plddt_func = plddt_func
        self.pos_func = pos_func
        self.form = form
        self.info = info_print
        self.pos_rule = pos_func_rule
        self.plddt_rule = plddt_func_rule
        self.logger = logger 
        self.save_dirs = save_dirs
    
    def get_loss(self, plddt, pos, ignore=None, job_name=None):
        
        info = []
        sum_max_loss = 0
        if ignore is None: ignore = []
        plddt_loss, pos_loss = [], []
        for sub_loss in self.plddt_func:
            if sub_loss.__class__.__name__ in ignore: continue
            sum_max_loss += sub_loss.max_loss
            try:
                sub_plddt_loss, sub_info_plddt = sub_loss.calculate_loss(plddt, job_name, self.save_dirs)
            except:
                sub_plddt_loss, sub_info_plddt = sub_loss.max_loss, {'Error':'plddt loss'}
            plddt_loss.append(sub_plddt_loss)
            info.append(sub_info_plddt)
        plddt_loss = self.plddt_rule(plddt_loss) if len(plddt_loss) != 0 else 0
        
        #plddt = [np.mean(plddt), np.min(plddt)]
        
        for sub_loss in self.pos_func:
            if sub_loss.__class__.__name__ in ignore: continue
            sum_max_loss += sub_loss.max_loss
            try:
                sub_pos_loss, sub_info_pos = sub_loss.calculate_loss(pos, plddt, job_name, self.save_dirs)
            except:
                sub_pos_loss, sub_info_pos = sub_loss.max_loss, {'Error':'pos loss'}
            pos_loss.append(sub_pos_loss)
            info.append(sub_info_pos)
        pos_loss = self.pos_rule(pos_loss) if len(pos_loss) !=0 else 0        
            
        loss = self.form(plddt_loss, pos_loss)
        
        # Collect into 1 Info
        for sub_info in info[1:]:
            for k,v in sub_info.items():
                info[0][k] = v
               
        if self.info:
            self.logger.info('[LSLOG]: Following info returned: ' + str(info[0]) )
        return loss/sum_max_loss


    def callback(self, plddt, pos, job_name):
        
        for sub_loss in self.plddt_func:
            sub_loss.callback(plddt, job_name, self.save_dirs)
        for sub_loss in self.pos_func:
            sub_loss.callback(pos, job_name, self.save_dirs)
    
