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

from Hallucinator.modules.Predict_ESM import predict_esm
from Hallucinator.modules.Compute_RMSD import compute_rmsd
from Hallucinator.modules.Init_Sequence import init_sequence
from Hallucinator.modules.Select_Mutation import select_mutation
from Hallucinator.modules.Mutations_BLOSUM62 import mutation_blosum62

from Hallucinator.loss.Loss_Function import LossFunction

from Hallucinator.loss.Compute_Cavity_Volume_Loss import CavityVolumeLoss
from Hallucinator.loss.Compute_Local_Distance_Loss import LocalDistanceLoss
from Hallucinator.loss.Compute_Secondary_Ratio_Loss import SecondaryRatioLoss
from Hallucinator.loss.Compute_Contact_Density_Loss import ContactDensityLoss
from Hallucinator.loss.Compute_Cavity_Similarity_Loss import CavitySimilarityLoss
from Hallucinator.loss.Compute_Cavity_Containing_Loss import CavityContainingLoss
from Hallucinator.loss.Compute_Protein_Containing_Loss import ProteinContainingLoss
from Hallucinator.loss.Compute_Preserve_Structure_Loss import PreserveStructureLoss
from Hallucinator.loss.Compute_Truncated_Average_PLDDT_Loss import TruncatedAveragePLDDTLoss
from Hallucinator.loss.Compute_Molecule_Binding_Affinity_Loss import MoleculeBindingAffinityLoss
from Hallucinator.loss.Compute_Cavity_Containing_Flexible_Loss import CavityContainingFlexibleLoss



# Code Adapted: https://stackoverflow.com/questions/11232230/logging-to-two-files-with-different-settings


# This a logger that will print both to screen and a file
def setup_logger(name, log_file, level=logging.INFO):

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(logging.FileHandler(log_file))
    #logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


class Protein_History_MCMC_Logger():

    # Length: Total length of protein hallucination
    # Excluded_aas: a list of symbol. Special amino acid will be not added
    # Temp: a list [Max_temp, Min_temp, (Temp_parameter)]. Parameter is for the Adaptive Control
    # Step: maximum number of MCMC steps
    # Temp_Control: Adaptive or Linear
    # Free_Guess: If free_guess not zero, then init (Free_Guess) difference sequence and start with the best
    # Seqc_Provided: None or a list. [Parent Sequence, Target RMSD (could be None), Mutation ratio (<1)]
    # Preserve_resid: None or an array of residue ID that will not be mutated
    # Guess_Loss_Ignore: None or a list of loss_names that will not be used in free guess.
    # Parent_Structure_File: None or a pdb that contains parent structure. 
    # PLDDT_Loss: a list of difference losses related to Plddt
    # Pos_Loss: a list of difference losses related to structure
    # form_loss: a function of Total_Losss = f(Plddt_loss, Pos_loss). For example, lambda x,y:x+y or lambda x,y:x*y
    # Loss_info: Boolean. Whether or not print loss information each turn to the logger
    # Pos_Rule: a function that combines all pos_loss. For example, np.sum() or np.max() 
    # PLDDDT_Rule: a function that combine all plddt_loss.
    # Job_name: None or a string. Name should not contains '/' or '..' to avoid being understood as folder
    # server_online: True: use ESM Api, False, read local ESM

    def __init__(self, length, excluded_aas, temp, step, temp_control='Adatpive',
                 free_guess=None, seqc_provided=None, preserve_resid=None,
                 guess_loss_ignore=None, parent_structure_file=None,
                 plddt_loss=[], pos_loss=[], form_loss=[], loss_info=True,
                 pos_rule=np.sum, plddt_rule=np.sum, job_name=None,
                 server_online = True, save_dirs = './',
                 ):

        self.excluded_aas = excluded_aas
        self.save_dirs = save_dirs
        if job_name is None:
            self.job_name = str(uuid.uuid4())[:8]
        else:
            self.job_name = str(job_name)
        # If no job name given, assign a random name.

        os.makedirs(self.save_dirs + 'results/'+self.job_name)

        self.logger = setup_logger('Logger',
                                   self.save_dirs + 'results/'+self.job_name+'/mcmc.log')
        self.logger.info('[JOBID]: Start with job name %s.' % self.job_name)
        # Init Logger (printing)

        if not server_online:
            model = torch.load("esmfold.model")
            model.eval().cuda()
            model.set_chunk_size(64)
            self.model = model
            self.logger.info('[ESMMD]: Finish init ESM Model.')
        else:
            self.model = None
        # Init Model, ESM

        self.random = self.set_random_seed()
        self.preserve_resid = preserve_resid
        self.loss = LossFunction(plddt_loss, pos_loss, form_loss, loss_info,
                                 pos_rule, plddt_rule, logger=self.logger,
                                 save_dirs=self.save_dirs)
        # Init Loss

        seqc = self.prepare_sequence(length, excluded_aas, free_guess,
                                     seqc_provided, guess_loss_ignore,
                                     parent_structure_file)

        plddt, pos = self.predict_seq(seqc)
        loss = self.calculate_loss(plddt, pos)
        self.record_info_init(seqc, loss, temp, plddt, pos, step, temp_control)
        # Record all initial information

    # %% init a sequence from None, or readin known sequence
    def prepare_sequence(self, length, excluded_aas, free_guess, seqc_provided,
                         guess_loss_ignore, parent_structure_file=None):

        if seqc_provided is None:
            self.logger.info('[SECIT]: Initial Sequence Random Generated.')
            if free_guess is None or free_guess == 0:
                seqc = init_sequence(length, excluded_aas, random=self.random)
            else:
                seqc = self.free_guess(free_guess, guess_loss_ignore, length,
                                       excluded_aas)
            self.parr_seqc = None
            self.parr_stru = None
            self.hist_rmsd = None
            self.traj_rmsd = None
        else:
            seqc, reverse_rmsd, mutate_rate = seqc_provided[0], seqc_provided[1], seqc_provided[2]
            self.logger.info(
                '[SECIT]: Initial Sequence Provided with random mutate rate ' + str(mutate_rate*100) + '%')
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
            nmut = int(len(self.parr_seqc)*mutate_rate)
            rmsd = self.count_stru_difference()
            if reverse_rmsd is not None:
                while (rmsd <= reverse_rmsd) or (rmsd > reverse_rmsd + 2) :
                    if (rmsd <= reverse_rmsd): 
                        nmut += 1
                    else:
                        nmut -= 1
                    seqc = self.yield_new_seqc(nmut, fake_plddt)
                    plddt, pos = self.predict_seq(seqc)
                    self.curr_stru = pos[pos[:, 2] == 'CA']
                    rmsd = self.count_stru_difference()
            elif free_guess is None or free_guess == 0:
                seqc = seqc
                plddt, pos = self.predict_seq(seqc)
                self.curr_stru = pos[pos[:, 2] == 'CA']
                rmsd = self.count_stru_difference()   
            else:
                seqc = self.free_guess(free_guess, guess_loss_ignore, length,
                                       excluded_aas) 
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

    # %% Set a random seed that can repeat
    def set_random_seed(self, seed=None):

        if seed is None:
            seed = int(np.random.rand() * (2**32 - 1))
        self.logger.info('[RANDS]: Random seed set as ' + str(seed))
        np.random.seed(seed)
        return np.random

    # %% Record information during MCMC
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

    # %% Completely random guess sequences
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

    # %% Predict a sequence and return PDB(pos) and PLDDT(ca)
    def predict_seq(self, seq):
        plddt, pos = predict_esm(self.model, seq, to_file=1,
                                 file_name=self.save_dirs + 'results/'+self.job_name+'/temp')
        ca_index = pos[:, 2] == 'CA'

        return plddt[ca_index], pos

    # %% Calculate loss from all loss function. Ctrl+C raise KeyInterrupt to stop
    def calculate_loss(self, plddt, pos, ignore=None):
        try:
            loss = self.loss.get_loss(plddt, pos, ignore, self.job_name)
            return loss
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        # except:
        #    return 100

    # %% Yield a new sequence. Mutation select from PLDDT
    def yield_new_seqc(self, size, plddt):
        mutations = select_mutation(
            len(self.curr_seqc), size, plddt, factor=0.05, random=self.random,
            preserve_resid=self.preserve_resid)
        new_seqc = list(copy.copy(self.curr_seqc))
        for mutation in mutations:
            new_seqc[mutation] = mutation_blosum62(
                new_seqc[mutation], self.excluded_aas)
        new_seqc = ''.join(new_seqc)
        return new_seqc

    # %% Temperature controlling part
    def temperature_control(self,):

        if self.temp_control == 'Linear':
            self.curr_temp = self.maxi_temp + \
                (self.mini_temp-self.maxi_temp)*(self.curr_step/self.maxi_step)
        if self.temp_control == 'Adaptive':
            r1, r2 = self.temp_setting
            self.curr_temp = min(
                self.mini_temp*(1 + r1*self.r**r2), self.maxi_temp)

    # %% Do a single step of MCMC
    def mcmc_single_step(self):

        if np.mean(self.curr_pldd) <= 50:
            n_mut = 3
        elif np.mean(self.curr_pldd) >= 70:
            n_mut = 1
        else:
            n_mut = 2
        # Determine how many mutations

        new_seqc = self.yield_new_seqc(n_mut, self.curr_pldd)
        try:
            plddt, pos = self.predict_seq(new_seqc)
            new_loss = self.calculate_loss(plddt, pos)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            # Unpredicted Structure. 
            plddt = np.zeros_like(self.curr_pldd)
            pos = np.copy(self.curr_stru)
            new_loss = 1.0
        self.hist_seqc.append(new_seqc)
        self.hist_loss.append(new_loss)
        # Record new loss and seqc

        if new_loss < self.curr_loss:
            p = 1.0
            acdc, self.r = True, 0
        else:
            p = np.exp(-(new_loss-self.curr_loss)/self.curr_temp)
            acdc = self.random.choice([False, True], p=[1-p, p])
            self.r = self.r+1
        self.hist_acdc.append(acdc)
        # Accept or Deny

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
        # If have a reference structure, compare a RMSD and sequence difference

        acc_sign = '√' if acdc else '×'
        self.logger.info('[MCLOG]: Step: %s, New Loss: %.3f, Curr Loss: %.3f, Best Loss: %.3f, Accepted: %s with temp %.1f and prob %.3f' % (
            self.curr_step, new_loss, self.curr_loss, np.min(self.hist_loss), acc_sign, self.curr_temp/self.mini_temp, p))
        # Logger print

        self.temperature_control()
        # Change Temperature

        self.traj_loss.append(self.curr_loss)

    # %% The overall MCMC. Print every print_level a figure. If allow_convergence, then could stop before max steps
    def mcmc(self, print_level=100, allow_convergence=False):
        try:
            while self.curr_step <= self.maxi_step:
                self.mcmc_single_step()
                if not self.curr_step % print_level:
                    if self.parr_seqc is not None: self.count_seqc_difference()
                    predict_esm(self.model, self.curr_seqc, to_file=1, to_figure=1,
                                file_name=self.save_dirs + 'results/' + self.job_name + '/'+str(self.curr_step//print_level))
                if allow_convergence != None and self.curr_loss < allow_convergence:
                    self.logger.info('[ESMMD]: Logger raise convergence.')
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            # This is auto/manual-stop, Ctrl+C stop the progress. Might have delay to stop every loss.
            pass

        self.best_seqc = self.hist_seqc[np.argmin(self.hist_loss)]

        self.logger.info('#'*100)
        self.logger.info('> Final Sequence:')
        self.logger.info(self.best_seqc)
        plddt, pos = predict_esm(self.model, self.best_seqc, to_file=1, to_figure=1,
                                 file_name=self.save_dirs + 'results/' + self.job_name + '/Best')
        plddt = plddt[pos[:, 2] == 'CA']
        self.loss.callback(plddt, pos, self.job_name)
        self.calculate_loss(plddt, pos)
        self.output_history_figure()
        # Call for a figure 

        print('[JODID]: Job finished with id:' + self.job_name)

    # %% a function count difference in sequence space.
    def count_seqc_difference(self):

        diff_count = 0
        for s in (difflib.ndiff(self.curr_seqc, self.parr_seqc)):
            if s[0] == ' ':
                continue
            diff_count += 1
        diff_count /= 2
        self.logger.info('[DFLOG]: Difference compared to Target(Parent) Seqc %s' % (int(diff_count)))
        return int(diff_count)
    
    # %% a function count difference in structure space
    def count_stru_difference(self):
        
        rmsd = str(np.round(compute_rmsd(self.parr_stru[:, 6:9].astype(
            'float')[:], self.curr_stru[:, 6:9].astype('float')[:],), 2))
        self.logger.info('[DFLOG]: Difference compared to Target(Parent) RMSD %s' % (rmsd))
        return float(rmsd)

    # %% call to create a search history figure
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

        plt.savefig(self.save_dirs + 'results/' + self.job_name + '/Search.png')



# Example 1, Recover 1OW4 from 50% mutation. Init RMSD 4-6.
'''
if __name__ == '__main__':

    seqc = 'NSSTQSYKDAMGPLVRECMGSVSATEDDFKTVLNRNPLESRTAQCLLACALDKVGLISPEGAIYTGDDLMPVMNRLYGFNDFKTVMKAKAVNDCANQVNGAYPDRCDLIKNFTDCVRNSY'
    
    logger = Protein_History_MCMC_Logger(
        length=len(seqc), excluded_aas=['C'], temp=[.5, .01, (1e-9, 5)], step=1000,
        free_guess=0, seqc_provided=(seqc, 4.00, 0.5), temp_control='Adaptive', 
        guess_loss_ignore=['TruncatedAveragePLDDTLoss',
                           'MoleculeBindingAffinityLoss'],
        plddt_loss=[TruncatedAveragePLDDTLoss(20, 80)],
        pos_loss=[ContactDensityLoss(target_density=0.035, max_loss=2),
                  MoleculeBindingAffinityLoss('./molecules/1OW4_BindE.pdbqt',
                                              max_loss=5, plddt_activate_value=50,
                                              exhaustiveness=16, target_score=10,
                                              )],
        form_loss=lambda x, y: x+y,
        pos_rule=np.sum, plddt_rule=np.mean,
        loss_info=True,
        parent_structure_file='./special/1OW4/1OW4_OriginalPDB.pdb')

    logger.mcmc(print_level=10, allow_convergence=0)
    np.savez('./results/'+logger.job_name+'/RMSD.npz', rmsd = logger.hist_rmsd)
'''

# Example 2, Secondary Structure Ratio 
'''
if __name__ == '__main__':

    logger = Protein_History_MCMC_Logger(
        length=90, excluded_aas=['C'], temp=[.5, .01, (1e-9, 5)], step=10000,
        free_guess=10, seqc_provided=None, temp_control='Adaptive',
        guess_loss_ignore=['TruncatedAveragePLDDTLoss',
                           'MoleculeBindingAffinityLoss'],
        plddt_loss=[TruncatedAveragePLDDTLoss(20, 80)],
        pos_loss=[ContactDensityLoss(target_density=0.035, max_loss=2),
                           SecondaryRatioLoss((90, 0), (2.5,2.5), (1.5,1.5), max_loss=5),
                           LocalDistanceLoss((3, 87), target_distance=7, max_loss=2),
                           LocalDistanceLoss((5, 85), target_distance=7, max_loss=2)],
        form_loss=lambda x, y: x+y,
        pos_rule=np.sum, plddt_rule=np.mean,
        loss_info=True)

    logger.mcmc(print_level=100, allow_convergence=0)
'''

# Example 3, Evolve Protein 1OW4 and Preserve Binding Structure, 30% and RMSD 3-5.
'''
if __name__ == '__main__':

    seqc = 'NSSTQSYKDAMGPLVRECMGSVSATEDDFKTVLNRNPLESRTAQCLLACALDKVGLISPEGAIYTGDDLMPVMNRLYGFNDFKTVMKAKAVNDCANQVNGAYPDRCDLIKNFTDCVRNSY’

    preserve = np.array([7,35,77,87,113,117,120])

    logger = Protein_History_MCMC_Logger(
        length=len(seqc), excluded_aas=['C'], temp=[.5, .01, (1e-9, 5)], step=30000,
        free_guess=30, seqc_provided=(seqc, 3.0, 0.3), temp_control=‘Adaptive‘,    
        preserve_resid=preserve, guess_loss_ignore=['TruncatedAveragePLDDTLoss',
                                                    'MoleculeBindingAffinityLoss'],
        plddt_loss=[TruncatedAveragePLDDTLoss(20, 80)],
        pos_loss=[ContactDensityLoss(target_density=0.035, max_loss=2),
                  SecondaryRatioLoss((60,20),(2.5,2.5), max_loss=5),
                  PreserveStructureLoss(preserve, './special/1OW4/1OW4_ESMFold.pdb',
                                        max_loss=5, k=30)],
        form_loss=lambda x, y: x+y,
        pos_rule=np.sum, plddt_rule=np.mean,
        loss_info=True,
        parent_structure_file='./special/1OW4/1OW4_OriginalPDB.pdb')

    logger.mcmc(print_level=100, allow_convergence=0)
'''
 
