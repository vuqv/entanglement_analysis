import sys

import MDAnalysis as mda
import numpy as np

pdb = sys.argv[1] + ".pdb"
u = mda.Universe(pdb, pdb)
ca_atoms = u.select_atoms("name CA")
raw_positions = ca_atoms.positions
print(f"There are {len(raw_positions)} residues in {pdb}")

ave_position = 0.5 * (raw_positions[:-1, :] + raw_positions[1:, :])
bond_vectors = - (raw_positions[:-1, :] - raw_positions[1:, :])


def cal_dis(r1, r2):
    return np.linalg.norm(r1 - r2)


N = len(ave_position)
d = 9
# find a loop
final_G = 0
IDX_i1 = 0
IDX_i2 = 0

"""
THis loop is only consider j1=i2+1 (maximize the open terminal)
"""
print('final_G, IDX_i1, IDX_i2, j1, j2')
for i1 in range(0, N):
    for i2 in range(i1 + 3, N):
        dis_i1i2 = cal_dis(raw_positions[i1, :], raw_positions[i2, :])
        if dis_i1i2 < d:
            # print("contact: ",i1, i2)
            Gc_ij = 0.0
            for i in range(i1, i2):
                # loop at N ter, open-segment at C-ter
                j1 = i2 + 1
                j2 = N
                for j in range(j1, j2):
                    temp = np.dot((ave_position[i, :] - ave_position[j, :]) / (
                            np.linalg.norm(ave_position[i, :] - ave_position[j, :]) ** 3),
                                  np.cross(bond_vectors[i, :], bond_vectors[j, :]))
                    Gc_ij += temp

            Gc_ij = Gc_ij / (4 * np.pi)
            if final_G < np.abs(Gc_ij):
                final_G = np.abs(Gc_ij)
                IDX_i1 = i1
                IDX_i2 = i2
                print(f'{final_G : .2f} {IDX_i1} {IDX_i2} {j1} {j2}')

            Gn_ij = 0.0
            for i in range(i1, i2):
                # loop at C ter, open-segment at N-ter
                j1 = 0
                j2 = i1 - 1
                for j in range(j1, j2):
                    temp = np.dot((ave_position[i, :] - ave_position[j, :]) / (
                            np.linalg.norm(ave_position[i, :] - ave_position[j, :]) ** 3),
                                  np.cross(bond_vectors[i, :], bond_vectors[j, :]))
                    Gn_ij += temp

            Gn_ij = Gn_ij / (4 * np.pi)
            if final_G < np.abs(Gn_ij):
                final_G = np.abs(Gn_ij)
                IDX_i1 = i1
                IDX_i2 = i2
                print(f'{final_G : .2f} {IDX_i1} {IDX_i2} {j1} {j2}')
print("=========================")
print(f'{final_G:.2f} {IDX_i1} {IDX_i2}')
