# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 14:15:01 2023

@author: jhy
"""


# %% packages and functions
import numpy as np


def compute_euclidean_distances_matrix(x, y):
    x_square = np.expand_dims(np.einsum('ij,ij->i', x, x), axis=1)
    if x is y:
        y_square = x_square.T
    else:
        y_square = np.expand_dims(np.einsum('ij,ij->i', y, y), axis=0)
    distances = np.dot(x, y.T)
    # use inplace operation to accelerate
    distances *= -2
    distances += x_square
    distances += y_square
    # result maybe less than 0 due to floating point rounding errors.
    np.maximum(distances, 0, distances)
    if x is y:
        # Ensure that distances between vectors and themselves are set to large number.
        # This may not be the case due to floating point rounding errors.
        distances.flat[::distances.shape[0] + 1] = 1000.0
    return distances


def compute_dihedral(p0, p1, p2, p3):

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 = b1 / np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)

    return np.degrees(np.arctan2(y, x))


def compute_secondary_structure_ratio(pos, return_detail=False, helx_cutoff=6, parr_cutoff=4, anti_cutoff=4):
    # cutoff = shortest acceptable length

    # pos = pd.read_csv('protein.pdb', sep='\s+', header=None).to_numpy()
    ca = pos[pos[:, 2] == 'CA']
    ca = ca[:, 5:9].astype('float')

    # define all the pseudo centers
    pseudo_center = (ca[:-1, 1:4]+ca[1:, 1:4])/2

    # %% alpha helix
    # calculate the i and i+3 pseudo center distances
    dist_pseudo_center_0_3 = np.linalg.norm(
        pseudo_center[3:, :]-pseudo_center[:-3, :], axis=1)
    # calculate all the torsion angles
    torsion_angle = np.empty(len(pseudo_center)-3)
    for i in range(len(pseudo_center)-3):
        torsion_angle[i] = compute_dihedral(
            pseudo_center[i, :], pseudo_center[i+1, :], pseudo_center[i+2, :], pseudo_center[i+3, :])

    # assign alpha helices
    pos_a0 = 1 + np.argwhere((dist_pseudo_center_0_3 > 4.21) & (
        dist_pseudo_center_0_3 < 5.23) & (torsion_angle > 43.5) & (torsion_angle < 78.3)).flatten()
    pos_a1 = np.array([])  # to store the complemented residue ids
    i0 = 0  # to initialize to the first continuous block
    # to complement all the residues belonging to alpha helices
    for i in range(len(pos_a0)-1):
        if i <= len(pos_a0)-3:
            diff = pos_a0[i+1]-pos_a0[i]
            if diff == 1:
                continue
            elif 2 <= diff <= 3:
                pos_a1 = np.concatenate((pos_a1,
                                         pos_a0[i0:i+1], np.array([pos_a0[i]+j for j in range(1, diff)])))
                i0 = i+1  # move to the next block
            else:
                pos_a1 = np.concatenate((pos_a1,
                                         pos_a0[i0:i+1], np.array([pos_a0[i]+j for j in range(1, 4)])))
                i0 = i+1
        else:  # append the end block
            pos_a1 = np.concatenate((pos_a1,
                                     pos_a0[i0:], np.array([pos_a0[-1]+j for j in range(1, 4)])))

    # to mark each continuous block (with diff_a > 1)
    diff_a = pos_a1[1:] - pos_a1[:-1]
    # to identify the position of each continuous block
    argwhere_i_3 = np.concatenate(
        (np.array([-1]), np.argwhere(diff_a > 1), np.array([len(pos_a1)])), axis=None)
    # initialize the final array to record the alpha helix range
    pos_a = np.array([])

    # abandon the helices shorter than helx_cutoff
    for i in range(len(argwhere_i_3)-1):
        if argwhere_i_3[i+1] - argwhere_i_3[i] >= helx_cutoff:
            pos_a = np.append(
                pos_a, pos_a1[argwhere_i_3[i]+1:argwhere_i_3[i+1]+1])
    pos_a = pos_a.astype('int')

    # output
    if len(pos_a) == 0:
        helx_ratio = 0
        #print('There is no ?-helix in this protein.\n')
    else:
        helx_ratio = np.round(len(pos_a)/len(ca), 2)
        #print('The ?-helix percentage is: {:.1%}'.format(len(pos_a)/len(ca)))
        if return_detail:
            print('The a-helix ranges are: {}'.format(pos_a) + '\n')

    # %% beta sheet
    dist_pseudo_center_i_j = np.sqrt(compute_euclidean_distances_matrix(
        pseudo_center[:-5, :], pseudo_center[5:, :]))

    # parallel
    # look for distances between i' and j'
    idx_pb_i_j = np.argwhere((dist_pseudo_center_i_j[1:, 1:] > 2.58) & (
        dist_pseudo_center_i_j[1:, 1:] < 5.18))
    # pick all the elements above or on the diagonal (residue diff >= 5)
    true_pair_pb_i_j = np.repeat(
        idx_pb_i_j[:, 1]-idx_pb_i_j[:, 0], 2).reshape((-1, 2))
    true_pair_pb_i_j = np.extract(
        true_pair_pb_i_j >= 0, idx_pb_i_j).reshape((-1, 2))

    # look for distances between i'-1 and j'-1
    idx_pb_i_1_j_1 = np.argwhere((dist_pseudo_center_i_j[:-1, :-1] > 4.34)
                                 & (dist_pseudo_center_i_j[:-1, :-1] < 5.03))
    true_pair_pb_i_1_j_1 = np.repeat(
        idx_pb_i_1_j_1[:, 1]-idx_pb_i_1_j_1[:, 0], 2).reshape((-1, 2))
    true_pair_pb_i_1_j_1 = np.extract(
        true_pair_pb_i_1_j_1 >= 0, idx_pb_i_1_j_1).reshape((-1, 2))

    # pick the corresponding positions
    true_pair_pb = np.array([])
    for i in range(len(true_pair_pb_i_j)):
        for j in range(len(true_pair_pb_i_1_j_1)):
            if true_pair_pb_i_j[i, 0] == true_pair_pb_i_1_j_1[j, 0] and\
                    true_pair_pb_i_j[i, 1] == true_pair_pb_i_1_j_1[j, 1]:

                true_pair_pb = np.concatenate((true_pair_pb, np.array(
                    [true_pair_pb_i_j[i, 0]+1, true_pair_pb_i_j[i, 1]+6]), np.array([true_pair_pb_i_j[i, 0]+2, true_pair_pb_i_j[i, 1]+7])), axis=None)
    true_pair_pb = np.sort(np.unique(true_pair_pb))

    if len(true_pair_pb) == 0:
        #print('There is no parallel ?-sheet in this protein.\n')
        parr_ratio = 0
    else:
        # to mark each continuous block
        diff_pb = true_pair_pb[1:] - true_pair_pb[:-1]
        # to identify the position of each continuous block
        argwhere_diff_pb = np.concatenate(
            (np.array([-1]), np.argwhere(diff_pb > 1), np.array([len(true_pair_pb)])), axis=None)
        # initialize the final array to record the parallel beta sheet range
        pos_pb = np.array([])

        # abandon the pb shorter than parr_cutoff
        for i in range(len(argwhere_diff_pb)-1):
            if argwhere_diff_pb[i+1] - argwhere_diff_pb[i] >= parr_cutoff:
                pos_pb = np.append(
                    pos_pb, true_pair_pb[argwhere_diff_pb[i]+1:argwhere_diff_pb[i+1]+1])

        pos_pb = pos_pb.astype('int')

        # output
        parr_ratio = np.round(len(pos_pb)/len(ca), 2)

        #print('The parallel ?-sheet percentage is: {:.1%}'.format(true_pair_pb.size/len(ca)))
        if return_detail:
            print('The parallel β-sheet ranges are: {}'.format(pos_pb) + '\n')

    # anti-parallel
    # look for distances between i' and j'
    idx_apb_i_j = np.argwhere((dist_pseudo_center_i_j > 4.36)
                              & (dist_pseudo_center_i_j < 5.19))
    true_pair_apb_i_j = np.repeat(
        idx_apb_i_j[:, 1] - idx_apb_i_j[:, 0], 2).reshape((-1, 2))
    true_pair_apb_i_j = np.extract(
        true_pair_apb_i_j >= 0, idx_apb_i_j).reshape((-1, 2))
    # convert it to residues
    true_pair_apb_i_j[:, 0] += 1
    true_pair_apb_i_j[:, 1] += 6

    # look for distances between i'+1 and j'-1
    idx_apb_i_1_j_1 = np.argwhere((dist_pseudo_center_i_j[1:-1, 1:-1] > 4.16)
                                  & (dist_pseudo_center_i_j[1:-1, 1:-1] < 5.27))
    true_pair_apb_i_1_j_1 = np.repeat(
        idx_apb_i_1_j_1[:, 1] - idx_apb_i_1_j_1[:, 0], 2).reshape((-1, 2))
    # extract the residue diff >= 3
    true_pair_apb_i_1_j_1 = np.extract(
        true_pair_apb_i_1_j_1 >= -2, idx_apb_i_1_j_1).reshape((-1, 2))
    true_pair_apb_i_1_j_1[:, 0] += 1
    true_pair_apb_i_1_j_1[:, 1] += 8

    # look for distances between C? i+1 and j
    dist_ca_i_1_j = np.sqrt(
        compute_euclidean_distances_matrix(ca[1:-4, 1:], ca[5:, 1:]))
    idx_apb_ca_i_1_j = np.argwhere(
        (dist_ca_i_1_j > 1.42) & (dist_ca_i_1_j < 5.99))
    true_ca_pair_apb_i_1_j = np.repeat(
        idx_apb_ca_i_1_j[:, 1]-idx_apb_ca_i_1_j[:, 0], 2).reshape((-1, 2))
    # extract the residue diff >= 4
    true_ca_pair_apb_i_1_j = np.extract(
        true_ca_pair_apb_i_1_j >= 0, idx_apb_ca_i_1_j).reshape((-1, 2))
    true_ca_pair_apb_i_1_j[:, 0] += 1
    true_ca_pair_apb_i_1_j[:, 1] += 6
    # pick the corresponding positions
    true_pair_apb = np.array([])
    for i in range(len(true_pair_apb_i_j)):
        for j in range(len(true_pair_apb_i_1_j_1)):
            for k in range(len(true_ca_pair_apb_i_1_j)):
                if true_pair_apb_i_j[i, 0] == true_pair_apb_i_1_j_1[j, 0] == true_ca_pair_apb_i_1_j[k, 0] and\
                        true_pair_apb_i_j[i, 1] == true_pair_apb_i_1_j_1[j, 1] == true_ca_pair_apb_i_1_j[k, 1]:
                    true_pair_apb = np.concatenate((true_pair_apb, true_pair_apb_i_j[i, :], np.array(
                        [true_pair_apb_i_j[i, 0]+1, true_pair_apb_i_j[i, 1]-1])))
    true_pair_apb = np.sort(np.unique(true_pair_apb))

    if len(true_pair_apb) == 0:
        anti_ratio = 0
    else:
        # to mark each continuous block
        diff_apb = true_pair_apb[1:] - true_pair_apb[:-1]
        # to identify the position of each continuous block
        argwhere_diff_apb = np.concatenate(
            (np.array([-1]), np.argwhere(diff_apb > 1), np.array([len(true_pair_apb)])), axis=None)
        # initialize the final array to record the antiparallel beta sheet range
        pos_apb = np.array([])

        # abandon the apb shorter than anti_cutoff
        for i in range(len(argwhere_diff_apb)-1):
            if argwhere_diff_apb[i+1] - argwhere_diff_apb[i] - 2 >= anti_cutoff:
                pos_apb = np.append(
                    pos_apb, true_pair_apb[argwhere_diff_apb[i]+2:argwhere_diff_apb[i+1]])

        pos_apb = pos_apb.astype('int')

        # output
        anti_ratio = np.round(len(pos_apb)/len(ca), 2)
        #print('The anti-parallel ?-sheet percentage is: {:.1%}'.format(true_pair_apb.size/len(ca)))

        if return_detail:
            print('The anti-parallel β-sheet ranges are: {}'.format(pos_apb) + '\n')

    return (helx_ratio, parr_ratio, anti_ratio)
