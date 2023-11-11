
import torch
import numpy as np
import pandas as pd
from io import StringIO
import subprocess as spc

from Hallucinator.modules.Plot_Protein import plot_protein

def predict_esm(model, sequence, num_recycles=3,
                to_file=0, file_name='Result',
                to_figure=0, figure_dpi=600):

    if model is not None:
      with torch.no_grad():
          output = model.infer_pdb(sequence,
                                  num_recycles=num_recycles,
                                  residue_index_offset=75)
    else:
      result = spc.run(['curl', '-k','--data', sequence,
       'https://api.esmatlas.com/foldSequence/v1/pdb/'], capture_output=1)
      assert not result.returncode, "ESM Server can be achieved."
      if isinstance(result.stdout, bytes):
        output = result.stdout.decode('utf-8')
      else:
        output = result.stdout

    multiliner = output.split(sep='\n')
    simp_multiliner = []
    for line in multiliner:
        if 'PARENT' in line or 'TER' in line or 'END' in line:
          continue
        if 'REMARK' in line:
          continue
        if 'ATOM' in line:
          simp_multiliner.append(line)
    file_output = '\n'.join(simp_multiliner)
    fake_file = StringIO(file_output)
    # Adjust format for pandas reading

    prot_frame = pd.read_table(fake_file, sep='\s+', header=None)
    plddt = np.asarray(prot_frame.iloc[:,10])
    if model is None:
      plddt =  plddt*100
    pos = np.asarray(prot_frame.iloc[:,:9])

    if to_file or to_figure:
        with open(file_name+'.pdb', mode='w') as f:
            print(fake_file.getvalue(), file=f)
    if to_figure:
        plot_protein(plddt=plddt, pos=pos, dpi=figure_dpi,
                     save=file_name+'.png')
    return plddt,pos
