"""
The length of loop and open segment >=10 (i1-i2 and j1-j2>=10)
I don't know why authors used this criteria but just use this.
usage: python ent_calculation.py PDB_ID.pdb
"""
import itertools
import sys
import time

# import MDAnalysis as mda
import numpy as np
from numba import njit

# pdb = sys.argv[1]
i1, i2, var_j1, var_j2 = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
len_seg = 4  # predefine minimum segment length, |j-i| >=len_seg

@njit(fastmath=True)
def cal_gc_ij(_i1, _i2, _j1, _j2):
    # index of ave positions
    _gc_ij = 0.0
    for i in range(_i1, _i2):
        for j in range(_j1, _j2):
            _gc_ij += np.dot(R_diff_3d[i, j, :] / np.linalg.norm(R_diff_3d[i, j, :]) ** 3, dR_cross_3d[i, j, :])
    _gc_ij = _gc_ij / (4 * np.pi)
    return _gc_ij


raw_positions = np.loadtxt('system')

# Calculate average positions and bond vectors in eq (2,3)
ave_positions = 0.5 * (raw_positions[:-1, :] + raw_positions[1:, :])
bond_vectors = - (raw_positions[:-1, :] - raw_positions[1:, :])
N = len(ave_positions)
nAtoms = len(raw_positions)


pair_array = np.asarray(list(itertools.product(ave_positions, ave_positions)))
R_diff = pair_array[:, 0, :] - pair_array[:, 1, :]
R_diff_3d = R_diff.reshape(N, N, 3)

# cross product term
pair_array = np.asarray(list(itertools.product(bond_vectors, bond_vectors)))
dR_cross = np.cross(pair_array[:, 0, :], pair_array[:, 1, :])
dR_cross_3d = dR_cross.reshape(N, N, 3)

pair_array = np.asarray(list(itertools.product(raw_positions, raw_positions)))
Distance_diff = pair_array[:, 0, :] - pair_array[:, 1, :]
Distance_pair = np.linalg.norm(Distance_diff, axis=1).reshape(nAtoms, nAtoms)

# # final_G = 0
# res = cal_gc_ij(i1, i2, j1, j2)
# print(res)

final_G = 0
"""In for loop, we add 1 because the right value of range function in python does not count"""
# for (i1, i2) in [(i1, i2) for i1 in range(N - len_seg + 1) for i2 in range(i1 + len_seg, N + 1)]:
for (j1, j2) in [(j1, j2) for j1 in range(var_j1, var_j2, 1) for j2 in range(j1 + len_seg, var_j2)]:
    # if Distance_pair[i1, i2] < 9.0 and ((j1 < i1 and j2 < i1) or (j1 > i2 and j2 > i2)):
    res = cal_gc_ij(i1, i2, j1, j2)   
    if final_G < np.abs(res):
        
        final_G = np.abs(res)
        print(final_G, j1, j2)