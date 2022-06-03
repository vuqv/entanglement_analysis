#!/usr/bin/env python3
################################################################################################################
script_title='OP'
script_version=1.7
script_author='Ian Sitarik'
script_updatelog=f"""Update log for {script_title} version {script_version}

                   Date: 08.20.2021
                   Note: Started covertion of topology_anal codes

                   Date: 09.01.2021
                   Note: v1.5 reproduced paper. This version will include a separate feature to iterate through all loops

                   Date: 09.11.2021
                   note: implimented joblib for multiprocessing support ~O(|native contacts|)

                  """

################################################################################################################

import os
import sys
import numpy as np
import time
import datetime
#import pickle
#import argparse
import re
import itertools
import multiprocessing
import warnings
from MDAnalysis import *
from scipy.spatial.distance import pdist, squareform
from itertools import product, combinations
from itertools import chain as iterchain
import configparser
import MDAnalysis.analysis.rms
import parmed as pmd
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import mdtraj as md
from joblib import Parallel, delayed
import numba as nb

##################################################################################################################
### START argument parse ###
##################################################################################################################
if len(sys.argv) != 13:
    print(f'[1] = ref structure')
    print(f'[2] = outfile_basename')
    print(f'[3] = in_path')
    print(f'[4] = psf')
    print(f'[5] = start_frame')
    print(f'[6] = end_frame')
    print(f'[7] = frame_stride')
    print(f'[8] = S')
    print(f'[9] = num processors')
    print(f'[10] = change threshold')
    print(f'[11] = out_path')
    print(f'[12] = model')
    quit()


in_path=sys.argv[3]
print(f'in_path: {in_path}')
if in_path == None: print(f'\n{script_updatelog}\n'); sys.exit()
else: in_paths=[x.strip('\n').split(', ') for x in os.popen(f'ls -v {in_path}').readlines()]
for ipath in in_paths: print(f'{ipath}')

out_path=sys.argv[11]
print(f'out_path: {out_path}')
if out_path == None: print(f'\n{script_updatelog}\n'); sys.exit()

psf=sys.argv[4]
print(f'psf: {psf}')
if psf == None: print(f'\n{script_updatelog}\n'); sys.exit()

print_updatelog=True
print(f'print_updatelog: {print_updatelog}')
if print_updatelog != None: print(f'\n{script_updatelog}\n')

global print_framesummary
print_framesummary=True
print(f'print_framesummary: {print_framesummary}')
if print_framesummary == None: print(f'\n{script_updatelog}\n')

outfile_basename=sys.argv[2]
print(f'outfile_basename: {outfile_basename}')
if outfile_basename == None: print(f'\n{script_updatelog}\n'); sys.exit()

start_frame=sys.argv[5]
print(f'start_frame: {start_frame}')
if start_frame == None:
    print(f'\n{script_updatelog}\n')
    start_frame = 0
else: start_frame=int(start_frame)

end_frame=sys.argv[6]
print(f'end_frame: {end_frame}')
if end_frame == None:
    print(f'\n{script_updatelog}\n')
    end_frame = 999999999999
else: end_frame=int(end_frame)

frame_stride=sys.argv[7]
print(f'frame_stride: {frame_stride}')
if frame_stride == None:
    print(f'\n{script_updatelog}\n')
    frame_stride = 1
else: frame_stride=int(frame_stride)

global S
S = int(sys.argv[8])
print(f'S: {S}')

global nproc
nproc = int(sys.argv[9])
print(f'nproc: {nproc}')

global change_threshold
change_threshold = float(sys.argv[10])
print(f'change_threshold: {change_threshold}')


orig_aa_pdb=sys.argv[1]
print(f'orig_aa_pdb: {orig_aa_pdb}')
if orig_aa_pdb == None: print(f'\n{script_updatelog}\n'); sys.exit()

model=sys.argv[12]
print(f'model: {model}')
if model == None: print(f'\n{script_updatelog}\n'); sys.exit()



##################################################################################################################
### START initial loading of structure files and qualtiy control ###
##################################################################################################################


### START dir declaration ###

if os.path.exists(f'{out_path}/'):
    print(f'{out_path}/ does exists and will be used')
    pass
else:
    os.makedirs(f'{out_path}/')

if os.path.exists(f'{out_path}{script_title}_{script_version}/'):
    print(f'{out_path}{script_title}_{script_version}/ does exists and will be used')
    pass
else:
    os.makedirs(f'{out_path}{script_title}_{script_version}/')

if os.path.exists(f'{out_path}{script_title}_{script_version}/logs/'):
    print(f'{out_path}{script_title}_{script_version}/logs/ does exists and will be used')
    pass
else:
    os.makedirs(f'{out_path}{script_title}_{script_version}/logs/')

if os.path.exists(f'{out_path}{script_title}_{script_version}/output/'):
    print(f'{out_path}{script_title}_{script_version}/output/ does exists and will be used')
    pass
else:
    os.makedirs(f'{out_path}{script_title}_{script_version}/output/')

### END dir declaration ###

### START preference setting ###

start_time=time.time() #time since epoch
print('time since epoch = '+str(start_time))

now=datetime.datetime.now() #time now at this moment
print('time now at this moment = '+str(now))
np.set_printoptions(threshold=sys.maxsize,linewidth=200)  #print total numpy matrix
np.set_printoptions(linewidth=200)  #print total numpy matrix
np.seterr(divide='ignore')

### END preference setting ###

######################################################################################################################
# USER DEFINED FUNCTIONS                                                                                             #
######################################################################################################################

def gen_nc_gdict(coor, coor_cmap, **kwargs):
    dom_nc_gdict = {}
    dom_gn_dict = {}
    dom_contact_ent = {}
    global dot_matrix, l

    #Nterm_thresh = termini_threshold[0]
    #Cterm_thresh = termini_threshold[1]
    #print(f'\ngen_nc_gdict analysis')
    #print(f'udom_pairs: {udom_pairs}')

    nc_indexs = np.stack(np.nonzero(coor_cmap)).transpose()

    l = len(coor)
    print(f'l: {l}')

    #make R and dR waves of length N-1
    range_l = np.arange(0, l-1)
    range_next_l = np.arange(1,l)

    R = 0.5*(coor[range_l] + coor[range_next_l])
    dR = coor[range_next_l] - coor[range_l]

    #make dRcross matrix
    pair_array = np.asarray(list(itertools.product(dR,dR)))

    x = pair_array[:,0,:]
    y = pair_array[:,1,:]

    dR_cross = np.cross(x,y)

    #make Rnorm matrix
    pair_array = np.asarray(list(itertools.product(R,R)))

    diff = pair_array[:,0,:] - pair_array[:,1,:]
    Runit = diff / np.linalg.norm(diff, axis=1)[:,None]**3
    Runit = Runit.astype(np.float32)

    #make final dot matrix
    dot_matrix = [np.dot(x,y) for x,y in zip(Runit,dR_cross)]
    dot_matrix = np.asarray(dot_matrix)
    dot_matrix = dot_matrix.reshape((l-1,l-1))

    #contact_ent = np.asarray([g_calc(i, j) for i,j in nc_indexs])
    #contact_ent = np.asarray([g_calc(i, j) for i,j in nc_indexs])
    #contact_ent = np.asarray(Parallel(n_jobs=nproc, )(delayed(g_calc)(i, j) for i,j in nc_indexs))
    contact_ent = np.asarray(Parallel(n_jobs=nproc)(delayed(g_calc)(i, j) for i,j in nc_indexs if j >= i + 10))

    # handel the situation where no native contacts have been formed yet
    if contact_ent.size == 0:
        contact_ent = np.asarray([[framenum, frametime, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    #outfilename = f'{out_path}{script_title}_{script_version}/output/{outfile_basename}_{framenum}_g_nc_loop_threads'
    #np.save(outfilename, contact_ent)
    #print(f'Saved: {outfilename}')
    max_contact_ent = max(contact_ent[:,4])
    maxarg = np.argmax(contact_ent[:,4])
    #contact_ent = 0
    #contact_ent = max(g_calc(i, j) for i,j in nc_indexs)
    print(f'max_contact_ent: {max_contact_ent} for S: {S} PDB: {in_paths[0][0]}')

    return (contact_ent[maxarg],contact_ent)


def g_calc(i,j):
    loop_range = np.arange(i,j)
    nterm_range = np.arange(0,i-S)
    gn_pairs = sub_lists(nterm_range,loop_range)
    if gn_pairs:
        #gn = max([abs(dot_matrix[p[:,0], p[:,1]].sum()/(4.0*np.pi)) for p in pairs])
        gn = np.asarray([helper_func(dot_matrix[p[:,0], p[:,1]]) for p in gn_pairs])
        #get thread j1 and j2
        gn_max_idx = np.argmax(gn)
        gn_j1 = min(gn_pairs[gn_max_idx][:,0])
        gn_j2 = max(gn_pairs[gn_max_idx][:,0])
        gn_ent = gn_pairs[gn_max_idx][np.argmax(dot_matrix[gn_pairs[gn_max_idx][:,0], gn_pairs[gn_max_idx][:,1]]),0]
        gn_max_pair = max(gn_pairs, key=len)
        gn_parital_link = dot_matrix[gn_max_pair[:,0], gn_max_pair[:,1]].sum()/(4.0*np.pi)

    else:
        gn = np.asarray([0])
        gn_parital_link = 0

    cterm_range = np.arange(j+S,l-1)

    gc_pairs = sub_lists(cterm_range,loop_range)
    if gc_pairs:
        #gc = max([abs(dot_matrix[p[:,0], p[:,1]].sum()/(4.0*np.pi)) for p in pairs])
        gc = np.asarray([helper_func(dot_matrix[p[:,0], p[:,1]]) for p in gc_pairs])

        #get thread j1 and j2
        gc_max_idx = np.argmax(gc)
        gc_j1 = min(gc_pairs[gc_max_idx][:,0])
        gc_j2 = max(gc_pairs[gc_max_idx][:,0])
        gc_ent = gc_pairs[gc_max_idx][np.argmax(dot_matrix[gc_pairs[gc_max_idx][:,0], gc_pairs[gc_max_idx][:,1]]),0]
        gc_max_pair = max(gc_pairs, key=len)
        gc_parital_link = dot_matrix[gc_max_pair[:,0], gc_max_pair[:,1]].sum()/(4.0*np.pi)

    else:
        gc = np.asarray([0])
        gc_parital_link = 0

    g_array = [max(gn), max(gc)]
    #total_link = round(gn_parital_link) + round(gc_parital_link)
    g = max(g_array)
    thread_id = np.argmax([abs(g_array[0]), abs(g_array[1])])

    if thread_id == 0 and np.sum(g_array) > 0:
        out = [framenum, frametime, gn_parital_link, gc_parital_link, g, i, j, gn_j1, gn_j2, gn_ent, (gn_ent)/(i)]
        return out

    elif thread_id == 1 and np.sum(g_array) > 0:
        out = [framenum, frametime, gn_parital_link, gc_parital_link, g, i, j, gc_j1, gc_j2, gc_ent, (l-gc_ent-1)/(l-j-1)]
        return out

    else:
        out = [framenum, frametime, 0, 0, 0, i, j, 0, 0, 0, 0]
        return out



#@nb.njit(fastmath=True)
def helper_func(g_vals: np.ndarray):

    return abs(g_vals.sum()/(4.0*np.pi))


def sub_lists(thread, loop):
    subs = []
    for i in range(len(thread)):
        for n in range(i+1,len(thread)+1):
            if abs(n - i) >= 10:
                sub = thread[i:n]
                subs.append(sub)

    return [ np.fromiter(itertools.chain(*itertools.product(row, loop)), int).reshape(-1, 2) for row in subs  ]


def cmap(cor, ref = True, restricted = True, cut_off = 9.0, bb_buffer = 4, **kwargs):
    if print_framesummary: print(f'\nCMAP generator')
    if print_framesummary: print(f'ref: {ref}\nrestricted: {restricted}\ncut_off: {cut_off}\nbb_buffer: {bb_buffer}')
    #make cmap for each domain and total structure

    distance_map=squareform(pdist(cor,'euclidean'))
    distance_map=np.triu(distance_map,k=1)


    contact_map = np.where((distance_map < 9) & (distance_map > 0), 1, 0)

    contact_num=contact_map.sum()

    if print_framesummary: print(f'Total number of contacts in is {contact_num}')

    return contact_map

def change_in_ent(frame_cont_ent_array, ref_cont_ent_array):
        chng_ent_dict={0: [], 1: [], 2: [], 3: [], 4: []}

        for nc in frame_cont_ent_array:
            #print(nc)
            frame_g = round(nc[2]) + round(nc[3])

            if any(np.equal(ref_cont_ent_array[:,5:7],[nc[5],nc[6]]).all(1)):
                ref_nc = ref_cont_ent_array[np.where((ref_cont_ent_array[:,5] == nc[5]) & (ref_cont_ent_array[:,6] == nc[6]))][0]

            else:
                continue

            ref_g = round(ref_nc[2]) + round(ref_nc[3])

            #check if diff in partial linking numbers is larger than change_threshold
            if (abs(nc[2] - ref_nc[2]) > change_threshold) or (abs(nc[3] - ref_nc[3]) > change_threshold):
                #nc_ij = np.vstack((ref_nc, nc))
                nc_ij = np.asarray([framenum,frametime,ref_g,frame_g])
                if abs(nc[4]) > abs(ref_nc[4]):
                    nc_ij = np.hstack((nc_ij,nc[5:]))
                elif abs(ref_nc[4]) > abs(nc[4]):
                    nc_ij = np.hstack((nc_ij,ref_nc[5:]))


                #if (ref_g != frame_g) and (abs(ref_nc[3] - nc[3]) > change_threshold):
                if (ref_g != frame_g):
                    if (abs(frame_g) > abs(ref_g)) and (frame_g*ref_g >= 0):
                        chng_ent_dict[0]+=[nc_ij]
                    elif (abs(frame_g) > abs(ref_g)) and (frame_g*ref_g < 0):
                        chng_ent_dict[1]+=[nc_ij]
                    elif (abs(frame_g) < abs(ref_g)) and (frame_g*ref_g >= 0):
                        chng_ent_dict[2]+=[nc_ij]
                    elif (abs(frame_g) < abs(ref_g)) and (frame_g*ref_g < 0):
                        chng_ent_dict[3]+=[nc_ij]
                    elif (abs(frame_g) == abs(ref_g)) and (frame_g*ref_g < 0):
                        chng_ent_dict[4]+=[nc_ij]

        return chng_ent_dict


######################################################################################################################
# MAIN                                                                                                               #
######################################################################################################################

### START loading of analysis universe ###
print('\n########################################START loading of analysis universe########################################\n')
#get alpha carbons atoms and then positions of them
print(f'Loading: {psf} & {in_paths}')
u = Universe(psf,in_paths)
print(model)
if model == 'aa':
    u_calphas = u.select_atoms('name CA')
elif model == 'cg':
    u_calphas = u.select_atoms('all')
print(u)
print(u_calphas)

global framenum,frametime
framenum = 0
frametime = 0.0
#load ref structs and get entanglement
if orig_aa_pdb != 'nan':
    print('\n########################################START loading of reference universe########################################\n')
    print(f'Loading: {orig_aa_pdb}')
    ref = Universe(orig_aa_pdb)
    print(ref)
    ref_calphas = ref.select_atoms('name CA')
    print(ref_calphas)
    ref_coor = ref_calphas.positions
    print(ref_coor[:10])

    ref_cmap = cmap(ref_coor)

    ref_dom_pair_contact_ent_dict = gen_nc_gdict(ref_coor, ref_cmap)

    #print(f'ref_dom_pair_contact_ent_dict: {ref_dom_pair_contact_ent_dict}')

### END loading of analysis universe ###

### START analysis of universe ###
outdata = []
outdata_labels = []
outdata_dtypes = []
print_framesummary
frame_times = []
print('\n########################################START analysis of trajectory########################################\n')

for ts in u.trajectory[start_frame:end_frame:frame_stride]:
    #if print_framesummary: print(f'\n\nFrame: {ts.frame}')

    framenum = ts.frame
    frametime = ts.time

    frame_start_time=time.time() #time since epoch

    frame_coor = u_calphas.positions

    #initilize output data structures and prime them with frame and time
    output = [framenum, frametime]
    if print_framesummary: print(f'\nframe_num, time: {output}')
    print(frame_coor[:10])

    frame_cmap = cmap(frame_coor)

    frame_dom_pair_contact_ent_dict = gen_nc_gdict(frame_coor, frame_cmap)

    #print(f'frame_dom_pair_contact_ent_dict: {frame_dom_pair_contact_ent_dict[0]}')

    #get change in ent
    if orig_aa_pdb != 'nan':
        change = change_in_ent(frame_dom_pair_contact_ent_dict[1],ref_dom_pair_contact_ent_dict[1])
        print(f'change:')
        outline = ['0','0','0','0','0']
        for k,v in change.items():
            if v:
                v = np.vstack(v)
                print(k,v[np.argmax(v[:,-1])])
                outline[k] = str(max(v[:,-1]))
        print(f'change_outline: {" ".join(outline)}')

        outfilename = f'{out_path}{script_title}_{script_version}/output/{outfile_basename}_{framenum}_changes'
        np.save(outfilename, change)
        print(f'Saved: {outfilename}')

    outdata.append(frame_dom_pair_contact_ent_dict[0])

    frame_times.append(time.time() - frame_start_time)

#post processes data
#add + 1 to resid
outdata = np.stack(outdata)
for e,o in enumerate(outdata):
    if o[5:10].sum() != 0:
        outdata[e,5:10] += 1

outfilename = f'{out_path}{script_title}_{script_version}/output/{outfile_basename}_g_timeseries'
np.save(outfilename, outdata)
np.savetxt(f'{outfilename}.txt', outdata, fmt='%i %1.4f %1.4f %1.4f %1.4f %i %i %i %i %i %1.4f', header=f'framenum, frametime, gn_parital_link, gc_parital_link, G_c, i1, i2, j1, j2, ent_res, depth')
print(f'Saved: {outfilename} and {outfilename}.txt')

#print(frame_times)
mean_frame_time = np.mean(frame_times)
print(f'mean_frame_time: {mean_frame_time}')

######################################################################################################################
comp_time=time.time()-start_time
print(f'computation time: {comp_time}')
print(f'NORMAL TERMINATION')
