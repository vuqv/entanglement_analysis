#!/usr/bin/env python3

import argparse
# import datetime
from  itertools import product, chain
import numpy as np
# import sys
from time import time
from warnings import filterwarnings
from MDAnalysis import *
#from functools import cache
from joblib import Parallel, delayed
from numba import njit
from scipy.spatial.distance import pdist, squareform
from memory_profiler import profile


filterwarnings("ignore")
"""
START argument parse
"""
parser = argparse.ArgumentParser(
    description="Gauss Entanglement calculations.\n Example: python gauss_linking.py -p TOP -f TRAJ -b 0 -e -1 -nt 1 "
                "-skip 1")
parser.add_argument('-top', '-p', type=str, help='Topology')
parser.add_argument('-traj', '-f', type=str, help='Trajectory')
parser.add_argument('-begin', '-b', type=int, help='Starting frame (default: 0)', default=0)
parser.add_argument('-end', '-e', type=int, help='End frame (default: last)', default=-1)
parser.add_argument('-skip', '-skip', type=int, help='analyze every nr-th frame', default=1)
parser.add_argument('-nproc', '-nt', type=int, help='number of processors to use', default=1)
parser.add_argument('-S', '-S', type=int, help='exclude residues before and after loop (default: 1)', default=1)
parser.add_argument('-outname', '-o', type=str, help='outfile basename')

args = parser.parse_args()

frame_skip = args.skip
# trajectory
in_paths = args.traj
psf = args.top
global outfile_basename
if args.outname:
    outfile_basename = args.outname
else:
    outfile_basename = args.traj.split('.')[0]
print(outfile_basename)
start_frame = args.begin
end_frame = args.end
# global S
S = args.S
print(f'Using {args.nproc} processor(s)')

"""START initial loading of structure files and quality control"""

start_time = time()  # time since epoch

np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(linewidth=100000)

# END preference setting


# USER DEFINED FUNCTIONS
@profile
def gen_nc_gdict(coor, coor_cmap):
    global dot_matrix, l

    nc_index = np.stack(np.nonzero(coor_cmap)).transpose()
    print(f'shape of nc_index: {nc_index.shape}')
    l = len(coor)
    print("shape of coor:", coor.shape)

    R = 0.5 * (coor[:-1:] + coor[1:, :])
    dR = coor[1:, :] - coor[:-1, :]

    # make dRcross matrix
    pair_array = np.asarray(list(product(dR, dR)))

    x = pair_array[:, 0, :]
    y = pair_array[:, 1, :]

    dR_cross = np.cross(x, y)

    del x, y

    # make Rnorm matrix
    pair_array = np.asarray(list(product(R, R)))

    diff = pair_array[:, 0, :] - pair_array[:, 1, :]
    Runit = diff / np.linalg.norm(diff, axis=1)[:, None] ** 3
    Runit = Runit.astype(np.float32)

    # make final dot matrix
    dot_matrix = [np.dot(x, y) for x, y in zip(Runit, dR_cross)]
    dot_matrix = np.asarray(dot_matrix)
    dot_matrix = dot_matrix.reshape((l - 1, l - 1))
    print('Dotmatrix complete')

    del diff, Runit, pair_array, dR_cross, R, dR

    contact_ent = np.asarray(Parallel(n_jobs=args.nproc)(delayed(g_calc)(i, j) for i, j in nc_index if j - i >= 10))
    print('contact_ent analysis complete')

    # handel the situation where no native contacts have been formed yet
    if contact_ent.size == 0:
        contact_ent = np.asarray([[framenum, frametime, 0, 0, 0, 0, 0]])

    del nc_index

    max_contact_ent = max(contact_ent[:, 2])
    maxarg = np.argmax(contact_ent[:, 2])
    print(f'max_contact_ent: {max_contact_ent:.3f} for PDB: {in_paths} {maxarg}')
    max_contact_ent = contact_ent[maxarg]
    del contact_ent

    return (max_contact_ent)


def g_calc(i, j):

    loop_range = np.arange(i, j)
    nterm_range = np.arange(0, i - S)

    if len(nterm_range) > 10:

        gn = 0
        gn_max = 0
        gn_j1 = 0
        gn_j2 = 0

        #p = np.fromiter(chain(*product(nterm_range, loop_range)), int).reshape(-1, 2)
        #partial_gn = helper_func(dot_matrix[p[:, 0], p[:, 1]])
        #if partial_gn > 0.6:
        for x in range(len(nterm_range)):
            for n in range(x + 1, len(nterm_range) + 1):
                if abs(n - x) >= 10:
                    sub = nterm_range[x:n]

                    p = np.fromiter(chain(*product(sub, loop_range)), int).reshape(-1, 2)
                    gn = helper_func(dot_matrix[p[:, 0], p[:, 1]])

                    if gn > gn_max:
                        gn_max = gn
                        gn_j1 = min(sub)
                        gn_j2 = max(sub)


    else:
        gn_max = 0

    del nterm_range

    cterm_range = np.arange(j + S, l - 1)
    if len(cterm_range) > 10:

        gc = 0
        gc_max = 0
        gc_j1 = 0
        gc_j2 = 0

        #p = np.fromiter(chain(*product(cterm_range, loop_range)), int).reshape(-1, 2)
        #partial_gc = helper_func(dot_matrix[p[:, 0], p[:, 1]])
        #if partial_gc > 0.6:
        for x in range(len(cterm_range)):
            for n in range(x + 1, len(cterm_range) + 1):
                if abs(n - x) >= 10:
                    sub = cterm_range[x:n]
                    p = np.fromiter(chain(*product(sub, loop_range)), int).reshape(-1, 2)
                    gc = helper_func(dot_matrix[p[:, 0], p[:, 1]])

                    if gc > gc_max:
                        gc_max = gc
                        gc_j1 = min(sub)
                        gc_j2 = max(sub)


    else:
        gc_max = 0

    del cterm_range

    g_array = [gn_max, gc_max]
    g = max(g_array)
    thread_id = np.argmax([abs(g_array[0]), abs(g_array[1])])

    if thread_id == 0 and np.sum(g_array) > 0:
        out = [framenum, frametime, gn_max, i, j, gn_j1, gn_j2]
        return out

    elif thread_id == 1 and np.sum(g_array) > 0:
        out = [framenum, frametime, gc_max, i, j, gc_j1, gc_j2]
        return out

    else:
        out = [framenum, frametime, 0, i, j, 0, 0]
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

    return [np.fromiter(chain(*product(row, loop)), int).reshape(-1, 2) for row in subs]


# @njit(fastmath=True, cache=True)
def cmap(cor, cut_off=9.0):
    distance_map = squareform(pdist(cor, 'euclidean'))
    distance_map = np.triu(distance_map, k=1)
    contact_map = np.where((distance_map < cut_off) & (distance_map > 0), 1, 0)
    # contact_num = contact_map.sum()

    del distance_map

    return contact_map

@profile
def outputdata(outdata):
    outdata = np.stack(outdata)
    print(outdata)

    for e, o in enumerate(outdata):
        if o[3:].sum() != 0:
            outdata[e, 3:] += 1

    outfile_name = f'{outfile_basename}_g_timeseries'
    np.save(outfile_name, outdata)
    np.savetxt(f'{outfile_name}.txt', outdata, fmt='%6i %10.3f %6.3f %5i %5i %5i %5i',
               header=f'frame, time, Max|G_c|, i1, i2, j1, j2')
    print(f'Saved: {outfile_name} and {outfile_name}.txt')
    return 'NORMAL TERMINATION'

"""
Main program
"""

print('\nSTART loading of analysis universe...\n')
# get alpha carbons atoms and then positions of them
print(f'Loading: {psf} & {in_paths}')
u = Universe(psf, in_paths)
print(u)
u_calphas = u.select_atoms('name CA')
#u_calphas = u.select_atoms('all')
print(u_calphas)
### START analysis of universe ###
outdata = []
# frame_times = []
n_frames = u.trajectory.n_frames
if end_frame == -1:
    end_frame = n_frames + 1
print(
    '\n########################################START analysis of trajectory########################################\n')
print(len(u.trajectory))
for ts in u.trajectory[start_frame:end_frame:frame_skip]:
    framenum = ts.frame
    frametime = ts.time

    # frame_start_time = time.time()  # time since epoch

    frame_coor = u_calphas.positions

    # initilize output data structures and prime them with frame and time
    output = [framenum, frametime]
    print(f'\nframe_num, time: {output}')

    frame_cmap = cmap(frame_coor)

    frame_dom_pair_contact_ent_dict = gen_nc_gdict(frame_coor, frame_cmap)
    print(frame_dom_pair_contact_ent_dict)
    outdata.append(frame_dom_pair_contact_ent_dict)

    # frame_times.append(time.time() - frame_start_time)

"""
Here we add 1 to loop and thread indeces.
If they are not 0- means the loop is existed.
add + 1 to resid i1,i2,j1, j2: which is in range (5:10)
"""

outputdata_flag = outputdata(outdata)

######################################################################################################################
comp_time = time() - start_time
print(f'computation time: {comp_time} seconds')
print(f'{outputdata_flag}')

