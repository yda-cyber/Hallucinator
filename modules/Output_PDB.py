#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 03:30:57 2023

@author: exouser
"""


def output_pdb(cd, atomname, filename='trace.pdb', mode='w+'):
    
    # Only suitable for ligand
    outfile = open(filename, mode)
    
    for index in range(len(cd)):
            
        if index<100:
            k,sep = 2, "    "
        elif index < 1000:
            k,sep = 0, "   "
        elif index < 10000:
            k,sep = 0, "  "
        else:
            k,sep = 0, " "
            
        atom = cd[index]
        name = atomname[index]
        outfile. write('ATOM'.ljust(6," ")+str(index+1).rjust(5," ")+" ")
        outfile. write(" "+name.ljust(3," "))
        outfile. write(" " + 'LIG' +" "+'L'+ "  " +
                   str(int(index)).rjust(k," ") + sep)
        outfile. write(str(round(atom[0],3)).rjust(8," ") + 
                   str(round(atom[1],3)).rjust(8," ")+ 
                   str(round(atom[2],3)).rjust(8," "))
        outfile. write("  1.00  0.00\n")    
    outfile. write("END\n")            