#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 05:23:27 2022

@author: Dai-Bei Yang


"""


import torch
import numpy as np
import pandas as pd
from io import StringIO

from modules.Plot_Protein import plot_protein


def predict_esm(model, sequence, num_recycles=3,
                to_file=0, file_name='Result',
                to_figure=0, figure_dpi=600):

    with torch.no_grad():
        output = model.infer_pdb(sequence,
                                 num_recycles=num_recycles,
                                 residue_index_offset=75)

    multiliner = output.split(sep='\n')
    simp_multiliner = []
    for line in multiliner:
        if 'PARENT' in line or 'TER' in line or 'END' in line:
            continue
        simp_multiliner.append(line)
    file_output = '\n'.join(simp_multiliner)
    fake_file = StringIO(file_output)
    # Adjust format for pandas reading

    prot_frame = pd.read_table(fake_file, sep='\s+', header=None)
    plddt = np.asarray(prot_frame.iloc[:,10])
    pos = np.asarray(prot_frame.iloc[:,:9])

    if to_file or to_figure:
        with open(file_name+'.pdb', mode='w') as f:
            print(fake_file.getvalue(), file=f)
    if to_figure:
        plot_protein(plddt=plddt, pos=pos, dpi=figure_dpi,
                     save=file_name+'.png')
    return plddt,pos


'''
torch.cuda.empty_cache()

# Simply Multchains
multimer = output.split(sep='\n')
simp_multimer = []
for line in multimer:
    if 'PARENT' in line or 'TER' in line or 'END' in line: continue
    simp_multimer.append(line)
output = '\n'.join(simp_multimer)

with open("result.pdb", "w") as f:
    f.write(output)

fig = plot_protein('result.pdb', dpi=1200)'''
