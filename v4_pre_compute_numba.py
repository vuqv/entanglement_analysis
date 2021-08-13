"""
This script follow what Kiet did:
i1 in range(N+1-10)
i2 in range(i1+10,N+1)
j1 in range(N+1-10)
j2 in range(j1+10,N+1)
Kiet used the smallest segment is 10 res.
what is different with v3 is in v4 we test on all posible combinations
"""
import itertools
import sys
import time

import MDAnalysis as mda
import numpy as np
from numba import jit

pdb = sys.argv[1] + ".pdb"

begin_time = time.time()

u = mda.Universe(pdb, pdb)
ca_atoms = u.select_atoms("name CA")
raw_positions = ca_atoms.positions
print(f"There are {len(raw_positions)} residues in {pdb}")

ave_positions = 0.5 * (raw_positions[:-1, :] + raw_positions[1:, :])
bond_vectors = - (raw_positions[:-1, :] - raw_positions[1:, :])
N = len(ave_positions)
nAtom = len(raw_positions)

pair_array = np.asarray(list(itertools.product(ave_positions, ave_positions)))
R_row_idx = pair_array[:, 0, :]
R_col_idx = pair_array[:, 1, :]
R_diff = R_row_idx - R_col_idx
R_diff_3d = R_diff.reshape(N, N, 3)

# [R(i)-R(j)]/norm[R(i)-R(j)]**3
pair_array = np.asarray(list(itertools.product(ave_positions, ave_positions)))
R_row_idx = pair_array[:, 0, :]
R_col_idx = pair_array[:, 1, :]
R_diff = (R_row_idx - R_col_idx)
R_diff_3d = R_diff.reshape(N, N, 3)

# cross produc term
pair_array = np.asarray(list(itertools.product(bond_vectors, bond_vectors)))
dR_row_idx = pair_array[:, 0, :]
dR_col_idx = pair_array[:, 1, :]
dR_cross = np.cross(dR_row_idx, dR_col_idx)
dR_cross_3d = dR_cross.reshape(N, N, 3)

pair_array = np.asarray(list(itertools.product(raw_positions, raw_positions)))

Distance_row_idx = pair_array[:, 0, :]
Distance_col_idx = pair_array[:, 1, :]
Distance_diff = Distance_row_idx - Distance_col_idx
Distance_pair = np.linalg.norm(Distance_diff, axis=1).reshape(nAtom, nAtom)

final_G = 0


@jit(nopython=True)
def cal_G_ij(i1, i2, j1, j2):
    G_ij = 0.0
    for i in range(i1, i2):
        for j in range(j1, j2):
            G_ij += np.dot(R_diff_3d[i, j, :] / np.linalg.norm(
                R_diff_3d[i, j, :]) ** 3, dR_cross_3d[i, j, :])
    G_ij = G_ij / (4 * np.pi)
    return G_ij, i1, i2, j1, j2


# i1,i2 loop to find native contact, distance < 9
for i1 in range(N - 1):
    for i2 in range(i1 + 1, N):
        for j1 in range(N - 1):
            for j2 in range(j1 + 1, N):
                if Distance_pair[i1, i2] < 9.0 and ((j1 < i1 and j2 < i1) or (j1 > i2 and j2 > i2)):
                    res = cal_G_ij(i1, i2, j1, j2)
                    if final_G < np.abs(res[0]):
                        final_G = np.abs(res[0])
                        IDX_i1 = res[1]
                        IDX_i2 = res[2]
                        IDX_j1 = res[3]
                        IDX_j2 = res[4]
                        print(
                            f'{final_G : .2f} {IDX_i1} {IDX_i2} {IDX_j1} {IDX_j2}')

end_time = time.time()
total_run_time = end_time - begin_time
print(f'Total execution time: {total_run_time / 60.0:.3f} mins')
