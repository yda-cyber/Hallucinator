#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 05:23:27 2022

@author: Dai-Bei Yang


"""

import os
import sys
import copy
import uuid
import torch
import logging
import difflib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

from modules.Predict_ESM import predict_esm
from modules.Compute_RMSD import compute_rmsd
from modules.Init_Sequence import init_sequence
from modules.Select_Mutation import select_mutation
from modules.Mutations_BLOSUM62 import mutation_blosum62

from loss.Loss_Function import LossFunction
from loss.Compute_Cavity_Volume_Loss import CavityVolumeLoss
from loss.Compute_Contact_Density_Loss import ContactDensityLoss
from loss.Compute_Cavity_Similarity_Loss import CavitySimilarityLoss
from loss.Compute_Cavity_Containing_Loss import CavityContainingLoss
from loss.Compute_Truncated_Average_PLDDT_Loss import TruncatedAveragePLDDTLoss
from loss.Compute_Molecule_Binding_Affinity_Loss import MoleculeBindingAffinityLoss


# Code Adapted: https://stackoverflow.com/questions/11232230/logging-to-two-files-with-different-settings


def setup_logger(name, log_file, level=logging.INFO):

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(logging.FileHandler(log_file))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


class Protein_History_MCMC_Logger():

    def __init__(self, length, excluded_aas, temp, step, temp_control='Adatpive',
                 free_guess=None, seqc_provided=None, guess_loss_ignore=None,
                 plddt_loss=[], pos_loss=[], form_loss=[], loss_info=True,
                 pos_rule=np.sum, plddt_rule=np.sum, job_name=None,
                 parent_structure_file=None):

        self.excluded_aas = excluded_aas

        if job_name is None:
            self.job_name = str(uuid.uuid4())[:8]
        else:
            self.job_name = str(job_name)

        os.makedirs('results/'+self.job_name)

        self.logger = setup_logger('Logger',
                                   './results/'+self.job_name+'/mcmc.log')
        self.logger.info('[JOBID]: Start with job name %s.' % self.job_name)

        model = torch.load("esmfold.model")
        model.eval().cuda()
        model.set_chunk_size(64)
        self.model = model
        self.logger.info('[ESMMD]: Finish init ESM Model.')
        # Init Model, ESM
        self.random = self.set_random_seed()
        self.loss = LossFunction(plddt_loss, pos_loss, form_loss, loss_info,
                                 pos_rule, plddt_rule, logger=self.logger)
        # Init Loss
        seqc = self.prepare_sequence(length, excluded_aas, free_guess,
                                     seqc_provided, guess_loss_ignore,
                                     parent_structure_file)

        plddt, pos = self.predict_seq(seqc)
        loss = self.calculate_loss(plddt, pos)
        self.record_info_init(seqc, loss, temp, plddt, pos, step, temp_control)

    def prepare_sequence(self, length, excluded_aas, free_guess, seqc_provided,
                         guess_loss_ignore, parent_structure_file=None):

        if seqc_provided is None:
            self.logger.info('[SECIT]: Initial Sequence Random Generated.')
            if free_guess is None:
                seqc = init_sequence(length, excluded_aas, random=self.random)
            else:
                seqc = self.free_guess(free_guess, guess_loss_ignore, length,
                                       excluded_aas)
            self.parr_seqc = None
            self.parr_stru = None
            self.hist_rmsd = None
            self.traj_rmsd = None
        else:
            seqc, reverse_rate = seqc_provided[0], seqc_provided[1]
            self.logger.info(
                '[SECIT]: Initial Sequence Provided with random reverse rate ' + str(reverse_rate) + '%')
            self.logger.info('[SECIT]: ' + seqc)
            plddt, pos = self.predict_seq(seqc)
            seqc = ''.join(seqc.split('\n'))
            fake_plddt = np.zeros(len(seqc)) + 100
            #fake_plddt[:100] = 0
            self.parr_seqc = seqc
            self.curr_seqc = seqc
            self.curr_stru = pos[pos[:,2] == 'CA']
            if parent_structure_file is None:
                self.parr_stru = pos[pos[:, 2] == 'CA']
            else:
                pos = pd.read_csv(parent_structure_file, sep='\s+', header=None).to_numpy()
                self.parr_stru = pos[pos[:, 2] == 'CA']
            nmut = int(len(self.parr_seqc)/2)
            rmsd = self.count_stru_difference()
            while (rmsd <= reverse_rate) or (rmsd > reverse_rate + 2) :
                if (rmsd <= reverse_rate): 
                    nmut += 1
                else:
                    nmut -= 1
                seqc = self.yield_new_seqc(nmut, fake_plddt)
                plddt, pos = self.predict_seq(seqc)
                self.curr_stru = pos[pos[:, 2] == 'CA']
                rmsd = self.count_stru_difference()
                
            '''
            n_mut = int(reverse_rate/100 * len(seqc))
            seqc = self.yield_new_seqc(n_mut, fake_plddt)
            '''

            self.curr_seqc = seqc
            plddt, pos = self.predict_seq(seqc)
            self.curr_stru = pos[pos[:, 2] == 'CA']
            self.count_seqc_difference()
            rmsd = self.count_stru_difference()
            self.hist_rmsd = [rmsd]
            self.traj_rmsd = [rmsd]
        return seqc

    def set_random_seed(self, seed=None):

        if seed is None:
            seed = int(np.random.rand() * (2**32 - 1))
        self.logger.info('[RANDS]: Random seed set as ' + str(seed))
        np.random.seed(seed)
        return np.random

    def record_info_init(self, seqc, loss, temp, plddt, pos, step, temp_control):

        temp_init, temp_fina, temp_setting = temp
        self.logger.info('[TEMMD]: ' + temp_control +
                         ' temperature control used with parameter ' + str(temp_setting))
        self.logger.info(
            '[TEMMD]: Following temperature parameter given (max,min) -> (%.5f, %.5f)' % (temp_init, temp_fina))
        self.hist_seqc = [seqc]
        self.hist_loss = [loss]
        self.hist_acdc = [True]
        self.traj_loss = [loss]

        self.curr_seqc = seqc
        self.curr_loss = loss
        self.curr_step = 0
        self.curr_pldd = plddt
        self.curr_stru = pos[pos[:, 2] == 'CA']
        self.maxi_step = step
        self.maxi_temp = temp_init
        self.mini_temp = temp_fina
        self.curr_temp = self.mini_temp if temp_control == 'Adaptive' else self.maxi_temp
        self.temp_control = temp_control
        self.temp_setting = temp_setting
        # Use for adaptive Temp control
        self.r = 0

    def free_guess(self, num, ignore, length, excluded_aas):

        self.hist_loss_guess = []
        self.hist_seqc_guess = []
        for i in range(num):
            seqc = init_sequence(length, excluded_aas, random=self.random)
            plddt, pos = self.predict_seq(seqc)
            plddt = np.zeros_like(plddt) + 100
            loss = self.calculate_loss(plddt, pos, ignore)
            self.hist_loss_guess.append(loss)
            self.hist_seqc_guess.append(seqc)
        return self.hist_seqc_guess[np.argmin(self.hist_loss_guess)]

    def predict_seq(self, seq):
        plddt, pos = predict_esm(self.model, seq, to_file=1,
                                 file_name='results/'+self.job_name+'/temp')
        ca_index = pos[:, 2] == 'CA'

        return plddt[ca_index], pos

    def calculate_loss(self, plddt, pos, ignore=None):
        try:
            loss = self.loss.get_loss(plddt, pos, ignore, self.job_name)
            return loss
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        # except:
        #    return 100

    def yield_new_seqc(self, size, plddt):
        mutations = select_mutation(
            len(self.curr_seqc), size, plddt, factor=0.05, random=self.random)
        new_seqc = list(copy.copy(self.curr_seqc))
        for mutation in mutations:
            new_seqc[mutation] = mutation_blosum62(
                new_seqc[mutation], self.excluded_aas)
        new_seqc = ''.join(new_seqc)
        return new_seqc

    def temperature_control(self,):

        if self.temp_control == 'Linear':
            self.curr_temp = self.maxi_temp + \
                (self.mini_temp-self.maxi_temp)*(self.curr_step/self.maxi_step)
        if self.temp_control == 'Adaptive':
            r1, r2 = self.temp_setting
            self.curr_temp = min(
                self.mini_temp*(1 + r1*self.r**r2), self.maxi_temp)

    def mcmc_single_step(self):

        if np.mean(self.curr_pldd) <= 50:
            n_mut = 3
        elif np.mean(self.curr_pldd) >= 70:
            n_mut = 1
        else:
            n_mut = 2

        new_seqc = self.yield_new_seqc(n_mut, self.curr_pldd)
        plddt, pos = self.predict_seq(new_seqc)
        new_loss = self.calculate_loss(plddt, pos)
        self.hist_seqc.append(new_seqc)
        self.hist_loss.append(new_loss)

        if new_loss < self.curr_loss:
            p = 1.0
            acdc, self.r = True, 0
        else:
            p = np.exp(-(new_loss-self.curr_loss)/self.curr_temp)
            acdc = self.random.choice([False, True], p=[1-p, p])
            self.r = self.r+1
        self.hist_acdc.append(acdc)

        if acdc:
            self.curr_loss = new_loss
            self.curr_seqc = new_seqc
            self.curr_pldd = plddt
            self.curr_stru = pos[pos[:, 2] == 'CA']

        self.curr_step = self.curr_step + 1

        if self.parr_seqc is not None:
            rmsd = self.count_stru_difference()
            self.hist_rmsd.append(rmsd)
            self.traj_rmsd.append(rmsd if acdc else self.traj_rmsd[-1])

        acc_sign = '√' if acdc else '×'
        self.logger.info('[MCLOG]: Step: %s, New Loss: %.3f, Curr Loss: %.3f, Best Loss: %.3f, Accepted: %s with temp %.1f and prob %.3f' % (
            self.curr_step, new_loss, self.curr_loss, np.min(self.hist_loss), acc_sign, self.curr_temp/self.mini_temp, p))

        self.temperature_control()
        self.traj_loss.append(self.curr_loss)

    def mcmc(self, print_level=100, allow_convergence=False):
        try:
            while self.curr_step <= self.maxi_step:
                self.mcmc_single_step()
                if not self.curr_step % print_level:
                    self.count_seqc_difference()
                    predict_esm(self.model, self.curr_seqc, to_file=1, to_figure=1,
                                file_name='results/' + self.job_name + '/'+str(self.curr_step//print_level))
                if self.curr_loss < 0.010 and allow_convergence:
                    self.logger.info('[ESMMD]: Logger raise convergence.')
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            # This is auto/manual-stop
            pass

        self.best_seqc = self.hist_seqc[np.argmin(self.hist_loss)]

        self.logger.info('#'*100)
        self.logger.info('> Final Sequence:')
        self.logger.info(self.best_seqc)
        plddt, pos = predict_esm(self.model, self.best_seqc, to_file=1, to_figure=1,
                                 file_name='results/' + self.job_name + '/Best')
        plddt = plddt[pos[:, 2] == 'CA']
        self.loss.callback(plddt, pos, self.job_name)
        self.calculate_loss(plddt, pos)
        self.output_history_figure()
        print('[JODID]: Job finished with id:' + self.job_name)

    def count_seqc_difference(self):

        diff_count = 0
        for s in (difflib.ndiff(self.curr_seqc, self.parr_seqc)):
            if s[0] == ' ':
                continue
            diff_count += 1
        diff_count /= 2
        self.logger.info('[DFLOG]: Difference compared to Target(Parent) Seqc %s' % (int(diff_count)))
        return int(diff_count)
        
    def count_stru_difference(self):
        
        rmsd = str(np.round(compute_rmsd(self.parr_stru[:, 6:9].astype(
            'float')[:], self.curr_stru[:, 6:9].astype('float')[:],), 2))
        self.logger.info('[DFLOG]: Difference compared to Target(Parent) RMSD %s' % (rmsd))
        return float(rmsd)

    def output_history_figure(self):

        plt.rcParams['figure.dpi'] = 1200
        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_subplot(111)
        ax.scatter(range(len(self.hist_loss)),
                   self.hist_loss, marker='*', s=5, alpha=.5)
        ax.plot(range(len(self.traj_loss)), self.traj_loss)
        ax.set_xlabel('MCMC Steps')
        ax.set_ylabel('Loss')
        ax.set_title('Searching Outcome')

        plt.savefig('results/' + self.job_name + '/Search.png')


if __name__ == '__main__':

    vdw = {}
    vdw['BEN'] = {'CG': 2.0, 'CD1': 2.0, 'CD2': 2.0, "CE2": 2.0, "CE1": 2.0, "CZ": 2.0,
                  'HG': 0.0, "HD1": 0.0, 'HD2': 0.0, "HE1": 0.0, "HE2": 0.0, 'HZ': 0.0}
    vdw['UNL'] = defaultdict(lambda: 1.5)

    seqc = 'NSSTQSYKDAMGPLVRECMGSVSATEDDFKTVLNRNPLESRTAQCLLACALDKVGLISPEGAIYTGDDLMPVMNRLYGFNDFKTVMKAKAVNDCANQVNGAYPDRCDLIKNFTDCVRNSY'
    #seqc = seqc[:110]

    logger = Protein_History_MCMC_Logger(
        length=len(seqc), excluded_aas=['C'], temp=[.5, .01, (1e-9, 5)], step=1000,
        free_guess=10, seqc_provided=(seqc, 4.00), temp_control='Adaptive',
        guess_loss_ignore=['TruncatedAveragePLDDTLoss',
                           'MoleculeBindingAffinityLoss'],
        plddt_loss=[TruncatedAveragePLDDTLoss(20, 80)],
        pos_loss=[ContactDensityLoss(target_density=0.035, max_loss=2),
                  MoleculeBindingAffinityLoss('./molecules/1OW4_BindE.pdbqt',
                                              max_loss=5, plddt_activate_value=50,
                                              exhaustiveness=16, target_score=10),
        # CavityVolumeLoss(0.3, 30, 0.01, 5)],
                 CavityContainingLoss('./molecules/1OW4_BindE.pdb', vdw,
                                    volume_factor=0.01, volume_expansion=500,
                                    similarity_factor=20, similarity_target_diff=0,
                                    sample_points=10, backbone_cavity=False,
                                    step=0.3, max_loss=5, plddt_activate_value=50)],
        #form_loss=lambda x, y: x+y,
        form_loss=lambda x, y: x+y,
        pos_rule=np.sum, plddt_rule=np.mean,
        loss_info=True,
        parent_structure_file='./special/1OW4/1OW4_OriginalPDB.pdb')

    logger.mcmc(print_level=10, allow_convergence=0)
    np.savez('./results/'+logger.job_name+'/RMSD.npz', rmsd = logger.hist_rmsd)
    # Use Hallucination to predict/understand mutations
