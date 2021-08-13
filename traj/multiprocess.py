 
import MDAnalysis as mda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import itertools
from multiprocessing import cpu_count

import multiprocessing
from multiprocessing import Pool
from functools import partial


# import dask
# import dask.multiprocessing
# from dask.distributed import Client

n_jobs = cpu_count()

len_seg = 10


def cal_gc_ij(R_diff_3d, dR_cross_3d, _i1, _i2, _j1, _j2):
    """
    This function calculates the double integral (4)
    Baiesi, M., Orlandini, E., Seno, F., & Trovato, A. (2017).
    Exploring the correlation between the folding rates of proteins and the entanglement of their native states.
    Journal of Physics A: Mathematical and Theoretical, 50(50). https://doi.org/10.1088/1751-8121/aa97e7
    :param R_diff_3d: (3D array) 3D array of Ri-Rj
    :param dR_cross_3d: (3D array) dRi-dRj
    :param _i1: (int) the first index of the loop
    :param _i2: (int) the second index of the loop
    :param _j1: (int) the first index of open segment
    :param _j2: (int) the second index of open segment
    :return: (float) |G|_{ij}
    """
    _gc_ij = 0.0
    for i in range(_i1, _i2):
        for j in range(_j1, _j2):
            _gc_ij += np.dot(R_diff_3d[i, j, :] / np.linalg.norm(R_diff_3d[i, j, :]) ** 3, dR_cross_3d[i, j, :])
    _gc_ij = _gc_ij / (4 * np.pi)
    return _gc_ij, _i1, _i2, _j1, _j2

def calculation_single_frame(frame_index, atomgroup):
    atomgroup.universe.trajectory[frame_index]

    raw_positions = atomgroup.positions
    # Calculate average positions and bond vectors in eq (2,3)
    ave_positions = 0.5 * (raw_positions[:-1, :] + raw_positions[1:, :])
    bond_vectors = - (raw_positions[:-1, :] - raw_positions[1:, :])
    N = len(ave_positions)
    nAtoms = len(raw_positions)

    """
    Precompute pair-wise matrix of R and dR
    when need to call, e.g Ri - Rj, just get element R_diff_3d[i,j,:]
    """

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

    """
     main loop, looking for all contact pairs (i_ loop) and 
     for each contact, looking for all segment possibility (j loop) and calculate Gij for that pair.
     If Gij > previous Gij, printout.
    """
    final_G = 0
    IDX_i1, IDX_i2, IDX_j1, IDX_j2 = None, None, None, None
    """In for loop, we add 1 because the right value of range function in python does not count"""
    for (i1, i2) in [(i1, i2) for i1 in range(N - len_seg + 1) for i2 in range(i1 + len_seg, N + 1)]:
        for (j1, j2) in [(j1, j2) for j1 in range(N - len_seg + 1) for j2 in range(j1 + len_seg, N + 1)]:
            if Distance_pair[i1, i2] <= 9.0 and ((j1 < i1 and j2 < i1) or (j1 > i2 and j2 > i2)):
                res = cal_gc_ij(R_diff_3d, dR_cross_3d, i1, i2, j1, j2)
                if final_G <= np.abs(res[0]):
                    final_G, IDX_i1, IDX_i2, IDX_j1, IDX_j2 = np.abs(res[0]), res[1], res[2], res[3], res[4]

    # return final_G, IDX_i1, IDX_i2, IDX_j1, IDX_j2
    return final_G


# @dask.delayed
# def analyze_block(blockslice, func, *args, **kwargs):
#     result = []
#     for ts in u.trajectory[blockslice.start:blockslice.stop]:
#         A = func(*args, **kwargs)
#         result.append(A)
#     return result



u = mda.Universe('ash1.pdb', '10frame.dcd')
ca_atoms = u.select_atoms("name CA")
resnames = ca_atoms.resnames
resids = ca_atoms.resids


run_per_frame = partial(calculation_single_frame,
                        atomgroup=ca_atoms,
                        )

frame_values = np.arange(u.trajectory.n_frames)
with Pool(n_jobs) as worker_pool:
    result = worker_pool.map(run_per_frame, frame_values)
result = np.asarray(result).T
print(result)

# n_frames = u.trajectory.n_frames
# n_blocks = n_jobs   #  it can be any realistic value (0 < n_blocks <= n_jobs)

# n_frames_per_block = n_frames // n_blocks
# blocks = [range(i * n_frames_per_block, (i + 1) * n_frames_per_block) for i in range(n_blocks-1)]
# blocks.append(range((n_blocks - 1) * n_frames_per_block, n_frames))





# jobs = []
# for bs in blocks:
#     jobs.append(analyze_block(bs,
#                               calculation_single_frame,
#                               ca_atoms))
# jobs = dask.delayed(jobs)
# results = jobs.compute()
# result = np.concatenate(results)
# print(result)
# if __name__ == '__main__':

#     client = Client(n_workers=n_jobs)
#     job_list = []
#     for frame in u.trajectory:
#         print(frame)
#         job_list.append(dask.delayed(calculation_single_frame(raw_positions=ca_atoms.positions)))

#     result = dask.compute(job_list)
#     print(result)