#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 02:10:48 2023

@author: exouser
"""

import numpy as np


def computeRotateMatrixKabsh(coordMat1, coordMat2):
    # coordMat1(True) <-(align)- coordMat2(Map) 
    
    coordMat1, coordMat2 = np.copy(coordMat1), np.copy(coordMat2)
    
    assert len(coordMat1) == len(coordMat2)
    
    m1 = np.mean(coordMat1, axis=0)
    m2 = np.mean(coordMat2, axis=0)
    
    coordMat1 -= m1
    coordMat2 -= m2
    
    H = np.matmul(coordMat2.T, coordMat1)
    
    u, s, vt = np.linalg.svd(H)
    
    v = vt.T
    
    d = np.linalg.det(v @ u.T)
    e = np.array([[1, 0, 0], [0, 1, 0], [0, 0, d]])
    
    r = v @ e @ u.T
    
    tt = m1 - np.matmul(r, m2)
    
    return r,tt


def alignMat(coordMat1, coordMat2):
    # coordMat1(True) <-(align)- coordMat2(Map) 
    
    r, tt = computeRotateMatrixKabsh(coordMat1, coordMat2)
    new_coordMat2 = []
    for i in coordMat2:
        point = np.matmul(r, i) + tt
        new_coordMat2.append(np.reshape(point, (1, 3)))
        
    new_coordMat2 = np.vstack(new_coordMat2)
    
    return new_coordMat2


def compute_rmsd(coordMat1, coordMat2, align=True):
    # Note this function do not align any weights
    
    if align:
        coordMat2 = alignMat(coordMat1, coordMat2)
        
    diff = coordMat1 - coordMat2
    return np.sqrt((diff * diff).sum() / coordMat1.shape[0])