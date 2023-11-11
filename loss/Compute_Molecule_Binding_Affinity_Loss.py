#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 22:40:18 2023

@author: exouser

Code Adapted from: 
    https://gitlab.com/miromoman/drepy/-/blob/master/DREPY.py
Online Openbabel Converter:
    http://www.cheminfo.org/Chemistry/Cheminformatics/FormatConverter/index.html
    # Remember to add Hydrogen and generate 3D structure
"""


import os
import shutil
import numpy as np
import pandas as pd
import subprocess as spc
import multiprocessing as mp
from matplotlib import pyplot as plt

from Hallucinator.modules.Output_PDB import output_pdb
from Hallucinator.loss.Compute_Contact_Density_Loss import compute_euclidean_distances_matrix


def model_check_residues_plddt(models, plddt, xyz_prot, affn):
    
    for i, model in enumerate(models):
        xyz_mol = model[1].to_numpy()[:, 6:9].astype('float')
        dist = compute_euclidean_distances_matrix(xyz_mol, xyz_prot)
        dist = np.sqrt(dist)
        near = [np.min(x) < 6 for x in dist.T]
        res_near_plddt = plddt[near]
        affn[i] = affn[i] * min(1,np.mean(res_near_plddt)/70)

    return affn


class MoleculeBindingAffinityLoss():

    def __init__(self, ligand_pdbqt, exhaustiveness=None, nmodes=10, ncpu=None,
                 max_loss=5, plddt_activate_value=50, use_effective_score=1,
                 target_score=15, box_resid=None):
        self.ligand_pdbqt = ligand_pdbqt
        if ncpu is None or ncpu == 0:
            ncpu = mp.cpu_count()
        self.ncpu = ncpu
        if exhaustiveness is None or exhaustiveness == 0:
            exhaustiveness = ncpu
        self.exhaustiveness = exhaustiveness
        self.nmodes = str(nmodes)
        self.plddt_activate_value = plddt_activate_value
        self.max_loss = max_loss
        self.use_effective_score = use_effective_score
        self.target_score = target_score
        self.box_resid = box_resid
    
    @staticmethod
    def convert_pdbqt_xyz(pdbqt, file_name, mode=0):
        
        with open(pdbqt) as dock:
            lines = dock.readlines()
        lines_ATOM = np.asarray(lines)[np.asarray(['ATOM' in l for l in lines])]
        df = pd.DataFrame(lines_ATOM, columns=['Data'])
        df = df.Data.str.split('\s+', expand=True)
        n_atom = int(df.iloc[-1][1])
        models = list(df.groupby(df.index // n_atom))       
        xyz_mol = models[mode][1].to_numpy()[:, 6:9].astype('float')
        name = models[mode][1].to_numpy()[:,2]
        
        output_pdb(xyz_mol, name, file_name)
        

    def calculate_loss(self, pos, plddt, job_name, capture_output=True, 
                       center=None, box=None):

        if np.mean(plddt) < self.plddt_activate_value:
            return self.max_loss, {}

        file = './results/'+job_name+'/temp.pdb'
        file_pdbqt = './results/'+job_name+'/temp.pdbqt'
        # Need ADFR ToolKits
        out = spc.run(["prepare_receptor", '-r', file, '-A', 'hydrogens', '-o', file_pdbqt],
                      capture_output=1)
        file_pdbqt = './results/'+job_name+'/temp.pdbqt'
        out_pdbqt = './results/'+job_name+'/dock.pdbqt'

        xyz = pos[:, 6:9].astype('float')
        xyz_ca = pos[pos[:, 2] == 'CA'][:, 6:9].astype('float')
        xyz_main = pos[pos[:, 2] == 'CA'][:, 6:9].astype('float')
        if self.box_resid is not None:
            xyz_ca = xyz_ca[self.box_resid-1]

        xmin, xmax = np.min(xyz_ca[:, 0]), np.max(xyz_ca[:, 0])
        ymin, ymax = np.min(xyz_ca[:, 1]), np.max(xyz_ca[:, 1])
        zmin, zmax = np.min(xyz_ca[:, 2]), np.max(xyz_ca[:, 2])

        if center is None:
            center = np.array([xmin+xmax, ymin+ymax, zmin+zmax])/2
        if box is None:
            box = np.array([xmax-xmin, ymax-ymin, zmax-zmin])+6

        center = np.round(center, 2).astype('str')
        box = np.round(box, 2).astype('str')

        try:
            out = spc.run(['qvinaw', '--receptor', file_pdbqt, '--ligand', self.ligand_pdbqt,
                           '--out', out_pdbqt, '--num_modes', str(
                               self.nmodes), '--cpu', str(self.ncpu),
                           '--exhaustiveness', str(self.exhaustiveness),
                           '--center_x', center[0], '--size_x', box[0],
                           '--center_y', center[1], '--size_y', box[1],
                           '--center_z', center[2], '--size_z', box[2]
                           ],
                          stderr=None, timeout=200, capture_output=capture_output)
            if capture_output:
                out = out.stdout.decode().split('\n')
                for i, line in enumerate(out):
                    if line == '-----+------------+----------+----------':
                        break
                # Look like this:  '   8         -7.7      4.966      9.680'
                df = pd.DataFrame(out, columns=['Affn'])
                df = df.iloc[i+1:-2]
                affn = df.Affn.str.split('\s+',expand=True).to_numpy()
                affn = affn[:, 2].astype('float')
      
                if self.use_effective_score:
                    # Split Dock File Models
                    with open(out_pdbqt) as dock:
                        lines = dock.readlines()
                    lines_ATOM = np.asarray(lines)[np.asarray(['ATOM' in l for l in lines])]
                    df = pd.DataFrame(lines_ATOM, columns=['Data'])
                    df = df.Data.str.split('\s+', expand=True)
                    n_atom = int(df.iloc[-1][1])
                    models = list(df.groupby(df.index // n_atom))
                    
                    affn = model_check_residues_plddt(models, plddt, xyz_main, affn)
                affn = np.min(affn)

            else:
                affn = 99
        except Exception as e:
            print(e)
            affn = 0  # Not found affn

        info_affn = {'Affinity': np.round(affn,2)}
        loss_affn = self.max_loss * min(1, max(0, (affn+self.target_score)/(0+self.target_score)))
        return loss_affn, info_affn

    def callback(self, pos, job_name):
        shutil.copyfile('./results/'+job_name+'/Best.pdb',
                        './results/'+job_name+'/temp.pdb')
        pass

        # What is the normal range of affinity (-20~0)

        '''
        fixer = PDBFixer('./results/'+job_name+'/temp.pdb')
        fixer.addMissingHydrogens(7)
        PDBFile.writeFile(fixer.topology, fixer.positions, open(file, 'w'))
        
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats('pdb', 'pdbqt')
        obConversion.AddOption('r')	
        
        mol = openbabel.OBMol()
        obConversion.ReadFile(mol, file) 
        obConversion.WriteFile(mol, './results/'+job_name+'/temp.pdbqt')
        '''

    def compute_average_affinity_ref50(self):
        
        print('[MBLRF]: Computing Average Binding Affinity for 50 Reference PDBs')
        print('[MBLRF]: This will take several minutes.')
        
        refdir = './molecules/Ref50PDBs/'
        pdbs = os.listdir(refdir)
        
        affns = []
        for pdb in pdbs:
            xyz = pd.read_csv(refdir+pdb, sep='\s+', skipfooter=1, 
                              header=None, engine='python').to_numpy()[:,5:8].astype('float')
            xmin, xmax = np.min(xyz[:, 0]), np.max(xyz[:, 0])
            ymin, ymax = np.min(xyz[:, 1]), np.max(xyz[:, 1])
            zmin, zmax = np.min(xyz[:, 2]), np.max(xyz[:, 2])
            center = np.array([xmin+xmax, ymin+ymax, zmin+zmax])/2
            box = np.array([xmax-xmin, ymax-ymin, zmax-zmin])+6
            
            center = np.round(center, 2).astype('str')
            box = np.round(box, 2).astype('str')
            
            out = spc.run(['qvinaw', '--receptor', refdir+pdb, '--ligand', self.ligand_pdbqt,
                           '--num_modes', str(self.nmodes), '--cpu', str(self.ncpu),
                           '--exhaustiveness', str(self.exhaustiveness),
                           '--center_x', center[0], '--size_x', box[0],
                           '--center_y', center[1], '--size_y', box[1],
                           '--center_z', center[2], '--size_z', box[2]
                           ],
                          stderr=None, timeout=1000, capture_output=1)
            try:
                out = out.stdout.decode().split('\n')
                for i, line in enumerate(out):
                    if line == '-----+------------+----------+----------':
                        break
                # Look like this:  '   8         -7.7      4.966      9.680'
                df = pd.DataFrame(out, columns=['Affn'])
                df = df.iloc[i+1:-2]
                affn = df.Affn.str.split('\s+',expand=True).to_numpy()
                affn = affn[:, 2].astype('float')[0]
            except:
                affn = 0
        
            affns.append(affn)
            
        affns = np.asarray(affns)
        mean = np.mean(affns[affns!=0])
        affns[affns==0] = mean
        plt.rcParams['figure.dpi'] = 1200
        fig = plt.figure(figsize=(2,3))
        plt.boxplot(affns)
        plt.ylabel('Binding Affinity')
        plt.xticks([1], ['Ref50 PDBs (Known)'])
        plt.title(self.ligand_pdbqt.split('/')[-1].split('.')[0])
        return affns