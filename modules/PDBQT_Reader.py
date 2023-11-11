#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 20:45:12 2023

@author: exouser
"""


import numpy as np
import pandas as pd

from io import StringIO

from modules.Output_PDB import output_pdb


class Branch():
    
    def __init__(self, data, link, init, end):
        
        self.data = data
        self.init = init
        self.link = link
        self.end = end
        self.sub_branch = []
        self.pnt_branch = None
        
    def __repr__(self):
        return ('Branch between <' + str(self.init) + ' ' + str(self.end) + '>, link to '+ str(self.link))

    def is_sub_branch(self, branch2):
        if self in branch2.sub_branch: return True
        
        if self.pnt_branch is None:
            if branch2.init <= self.init and branch2.end >= self.end and branch2 != self:
                self.pnt_branch = branch2 
                if self not in branch2.sub_branch:
                    branch2.sub_branch.append(self)
                return True    
            return False
        else:
            # Is already a sub_branch
            pnt_branch = self.pnt_branch
            if pnt_branch == branch2: return True
            flag1 = (branch2.init <= self.init and branch2.end >= self.end and branch2 != self)
            flag2 = (branch2.is_sub_branch(pnt_branch))
            if flag1 and flag2:
                pnt_branch.sub_branch.remove(self)
                if self not in branch2.sub_branch:
                    branch2.sub_branch.append(self)
                self.pnt_branch = branch2
                return True
            return False
        
    def find_atom_index(self, index):
        idx = list(self.data[:, 1])
        return self.data[idx.index(index)]
        
    def rotate_branch(self, angle):
        
        pass
    
    def synchronize(self):
        
        pass

class PDBQTReader():
    
    def __init__(self, pdbqt_file):
        
        self.pdbqt_file = pdbqt_file
        
        with open(pdbqt_file) as g: string = g.readlines()
        string = self.throw_remarks(string)
        self.root = self.read_root(string)
        self.branch = self.read_branch(string)
        

        pass
    
    @staticmethod
    def throw_remarks(string):
        new_string = []
        for line in string:
            if 'REMARK' in line: continue
            new_string.append(line)
        return new_string
    
    def read_root(self, string):
        root_string, f1, f2 = [], 0, 0
        for i, line in enumerate(string):
            if line == 'ROOT\n': f1 = i
            if line == 'ENDROOT\n': f2 = i
        root_string = string[f1+1:f2]
        root_string = ''.join(root_string)
        root_datafr = pd.read_table(StringIO(root_string), header=None, sep='\s+')
        root_datafr = root_datafr.fillna('NA').to_numpy()
        
        root_branch = Branch(root_datafr, (0,0), 0, np.inf)
        return root_branch
    
    def read_branch(self, string):
        i, branches, flag = 0, [], False
        while not flag:
           branch, j, k, flag = PDBQTReader.find_branch(string[i:])
           if flag: break
           branches.append(branch)
           i += (j+1)
         
        branch_tree = []
        for branch in branches:
            branch_string = []
            for line in branch[1:-1]:
                if 'ATOM' in line: branch_string.append(line)
            branch_string = ''.join(branch_string)
            branch_datafr = pd.read_table(StringIO(branch_string), header=None, sep='\s+')
            branch_datafr = branch_datafr.fillna('NA').to_numpy()
            branch_link = int(branch[0].split()[1]), int(branch[0].split()[2]),
            branch_init, branch_end = int(branch_datafr[0][1]), int(branch_datafr[-1][1])
            branch_tree.append(Branch(branch_datafr, branch_link, branch_init, branch_end))
        
        for br in branch_tree:
            for j in range(len(branch_tree)):
                br.is_sub_branch(branch_tree[j])
        
        for br in branch_tree:
            if br.pnt_branch is None:
                br.is_sub_branch(self.root)
                
        return branch_tree
    
    @staticmethod
    def find_branch(string):
        branch_string, f1, f2, n1, n2 = [], None, None, 0, 0
        for i, line in enumerate(string):
            if f1 is None:
                if 'BRANCH' in line and 'ENDBRANCH' not in line:
                    detail = line.split()
                    n1, n2 = detail[1], detail[2] 
                    f1 = i 
                    continue
            if f2 is None and f1 is not None:
                if 'ENDBRANCH' in line:
                    detail = line.split()
                    if n1 == detail[1] and n2 == detail[2]:
                        f2 = i
        if f1 is not None and f2 is not None:
            branch_string = string[f1 : f2+1]
            return branch_string, f1, f2, False
        else:
            return [], 0, 0, True
    
    def rotate_branch(self, branch, angle):
        
        pass
    
    
    def output_pdb(self, file_name):
        
        cd = list(self.root.data[:,5:8])
        an = list(self.root.data[:, 2])
        for branch in self.root.sub_branch:
            cd += list(branch.data[:,5:8])
            an += list(branch.data[:,2])
        cd = np.asarray(cd).astype('float')
        output_pdb(cd, an, file_name)
        
    
