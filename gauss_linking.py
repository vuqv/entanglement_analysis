#!/usr/bin/env python3

import argparse
import datetime
import itertools
import sys
import time

from numba import njit
from functools import cache
import numpy as np
from MDAnalysis import *
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist, squareform

import warnings
warnings.filterwarnings("ignore")
"""
START argument parse
"""
parser = argparse.ArgumentParser(
    description="Gauss Entanglement calculations.\n Example: python ent_calculation.py -p TOP -f TRAJ -b 0 -e -1")
parser.add_argument('-top', '-p', type=str, help='Topology')
parser.add_argument('-traj', '-f', type=str, help='Trajectory')
parser.add_argument('-begin', '-b', type=int, help='Starting frame (default: 0)', default=0)
parser.add_argument('-end', '-e', type=int, help='End frame (default: last)', default=-1)
parser.add_argument('-stride', '-stride', type=int, help='End frame (default: last)', default=1)
parser.add_argument('-nproc', '-nproc', type=int, help='number of processors to use', default=1)
parser.add_argument('-S', '-S', type=int, help='exclude residues before and after loop (default: 1)', default=1)

args = parser.parse_args()
frame_stride = args.stride

# trajectory
in_paths = args.traj

psf = args.top

outfile_basename = args.traj.split('.')[0]

start_frame = args.begin

end_frame = args.end

global S
S = args.S

global nproc

nproc = args.nproc
print(f'nproc: {nproc}')

"""START initial loading of structure files and qualtiy control"""
# START preference setting

start_time = time.time()  # time since epoch
# print('time since epoch = ' + str(start_time))

# now = datetime.datetime.now()  # time now at this moment
# print('time now at this moment = ' + str(now))
# np.set_printoptions(threshold=sys.maxsize, linewidth=200)  # print total numpy matrix
# np.set_printoptions(linewidth=200)  # print total numpy matrix
np.seterr(divide='ignore', invalid='ignore')


# END preference setting


# USER DEFINED FUNCTIONS
def gen_nc_gdict(coor, coor_cmap):
    # dom_nc_gdict = {}
    # dom_gn_dict = {}
    # dom_contact_ent = {}
    global dot_matrix, l

    nc_indexs = np.stack(np.nonzero(coor_cmap)).transpose()

    l = len(coor)

    # make R and dR waves of length N-1
    range_l = np.arange(0, l - 1)
    range_next_l = np.arange(1, l)

    R = 0.5 * (coor[range_l] + coor[range_next_l])
    dR = coor[range_next_l] - coor[range_l]

    # make dRcross matrix
    pair_array = np.asarray(list(itertools.product(dR, dR)))

    x = pair_array[:, 0, :]
    y = pair_array[:, 1, :]

    dR_cross = np.cross(x, y)

    # make Rnorm matrix
    pair_array = np.asarray(list(itertools.product(R, R)))

    diff = pair_array[:, 0, :] - pair_array[:, 1, :]
    Runit = diff / np.linalg.norm(diff, axis=1)[:, None] ** 3
    Runit = Runit.astype(np.float32)

    # make final dot matrix
    dot_matrix = [np.dot(x, y) for x, y in zip(Runit, dR_cross)]
    dot_matrix = np.asarray(dot_matrix)
    dot_matrix = dot_matrix.reshape((l - 1, l - 1))

    contact_ent = np.asarray(Parallel(n_jobs=nproc)(delayed(g_calc)(i, j) for i, j in nc_indexs if j >= i + 10))

    # handel the situation where no native contacts have been formed yet
    if contact_ent.size == 0:
        contact_ent = np.asarray([[framenum, frametime, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    max_contact_ent = max(contact_ent[:, 4])
    maxarg = np.argmax(contact_ent[:, 4])

    print(f'max_contact_ent: {max_contact_ent} for PDB: {in_paths}')

    return (contact_ent[maxarg], contact_ent)


def g_calc(i, j):
    loop_range = np.arange(i, j)
    nterm_range = np.arange(0, i - S)
    gn_pairs = sub_lists(nterm_range, loop_range)
    if gn_pairs:
        # gn = max([abs(dot_matrix[p[:,0], p[:,1]].sum()/(4.0*np.pi)) for p in pairs])
        gn = np.asarray([helper_func(dot_matrix[p[:, 0], p[:, 1]]) for p in gn_pairs])
        # get thread j1 and j2
        gn_max_idx = np.argmax(gn)
        gn_j1 = min(gn_pairs[gn_max_idx][:, 0])
        gn_j2 = max(gn_pairs[gn_max_idx][:, 0])
        gn_ent = gn_pairs[gn_max_idx][np.argmax(dot_matrix[gn_pairs[gn_max_idx][:, 0], gn_pairs[gn_max_idx][:, 1]]), 0]
        gn_max_pair = max(gn_pairs, key=len)
        gn_parital_link = dot_matrix[gn_max_pair[:, 0], gn_max_pair[:, 1]].sum() / (4.0 * np.pi)

    else:
        gn = np.asarray([0])
        gn_parital_link = 0

    cterm_range = np.arange(j + S, l - 1)

    gc_pairs = sub_lists(cterm_range, loop_range)
    if gc_pairs:
        # gc = max([abs(dot_matrix[p[:,0], p[:,1]].sum()/(4.0*np.pi)) for p in pairs])
        gc = np.asarray([helper_func(dot_matrix[p[:, 0], p[:, 1]]) for p in gc_pairs])

        # get thread j1 and j2
        gc_max_idx = np.argmax(gc)
        gc_j1 = min(gc_pairs[gc_max_idx][:, 0])
        gc_j2 = max(gc_pairs[gc_max_idx][:, 0])
        gc_ent = gc_pairs[gc_max_idx][np.argmax(dot_matrix[gc_pairs[gc_max_idx][:, 0], gc_pairs[gc_max_idx][:, 1]]), 0]
        gc_max_pair = max(gc_pairs, key=len)
        gc_parital_link = dot_matrix[gc_max_pair[:, 0], gc_max_pair[:, 1]].sum() / (4.0 * np.pi)

    else:
        gc = np.asarray([0])
        gc_parital_link = 0

    g_array = [max(gn), max(gc)]
    # total_link = round(gn_parital_link) + round(gc_parital_link)
    g = max(g_array)
    thread_id = np.argmax([abs(g_array[0]), abs(g_array[1])])

    if thread_id == 0 and np.sum(g_array) > 0:
        out = [framenum, frametime, gn_parital_link, gc_parital_link, g, i, j, gn_j1, gn_j2, gn_ent, (gn_ent) / (i)]
        return out

    elif thread_id == 1 and np.sum(g_array) > 0:
        out = [framenum, frametime, gn_parital_link, gc_parital_link, g, i, j, gc_j1, gc_j2, gc_ent,
               (l - gc_ent - 1) / (l - j - 1)]
        return out

    else:
        out = [framenum, frametime, 0, 0, 0, i, j, 0, 0, 0, 0]
        return out


@njit(fastmath=True, cache=True)
def helper_func(g_vals: np.ndarray):
    return abs(g_vals.sum() / (4.0 * np.pi))


# @njit(fastmath=True, cache=True)
def sub_lists(thread, loop):
    subs = []
    for i in range(len(thread)):
        for n in range(i + 1, len(thread) + 1):
            if abs(n - i) >= 10:
                sub = thread[i:n]
                subs.append(sub)

    return [np.fromiter(itertools.chain(*itertools.product(row, loop)), int).reshape(-1, 2) for row in subs]


# @njit(fastmath=True, cache=True)
def cmap(cor, cut_off=9.0):
    distance_map = squareform(pdist(cor, 'euclidean'))
    distance_map = np.triu(distance_map, k=1)
    contact_map = np.where((distance_map < cut_off) & (distance_map > 0), 1, 0)
    # contact_num = contact_map.sum()

    return contact_map


"""
Main program
"""
# START loading of analysis universe
print('\nSTART loading of analysis universe...\n')
# get alpha carbons atoms and then positions of them
print(f'Loading: {psf} & {in_paths}')
u = Universe(psf, in_paths)
u_calphas = u.select_atoms('name CA')

### START analysis of universe ###
outdata = []
frame_times = []
n_frames = u.trajectory.n_frames
if end_frame == -1:
    end_frame = n_frames + 1
print(
    '\n########################################START analysis of trajectory########################################\n')

for ts in u.trajectory[start_frame:end_frame:frame_stride]:
    framenum = ts.frame
    frametime = ts.time

    frame_start_time = time.time()  # time since epoch

    frame_coor = u_calphas.positions

    # initilize output data structures and prime them with frame and time
    output = [framenum, frametime]
    print(f'\nframe_num, time: {output}')

    frame_cmap = cmap(frame_coor)

    frame_dom_pair_contact_ent_dict = gen_nc_gdict(frame_coor, frame_cmap)

    outdata.append(frame_dom_pair_contact_ent_dict[0])

    frame_times.append(time.time() - frame_start_time)

# post processes data
# add + 1 to resid
outdata = np.stack(outdata)
for e, o in enumerate(outdata):
    if o[5:10].sum() != 0:
        outdata[e, 5:10] += 1

outfile_name = f'{outfile_basename}_g_timeseries'
np.save(outfile_name, outdata)
np.savetxt(f'{outfile_name}.txt', outdata, fmt='%6i %10.3f %6.3f %6.3f %6.3f %5i %5i %5i %5i %5i %6.4f',
           header=f'frame, time, gn, gc, Max|G_c|, i1, i2, j1, j2, ent_res, depth')
print(f'Saved: {outfile_name} and {outfile_name}.txt')

# print(frame_times)
# mean_frame_time = np.mean(frame_times)
# print(f'mean_frame_time: {mean_frame_time}')

######################################################################################################################
comp_time = time.time() - start_time
print(f'computation time: {comp_time}')
print(f'NORMAL TERMINATION')
