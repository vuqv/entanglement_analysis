import MDAnalysis as mda
import numpy as np
from scipy import spatial
import sys, time
import itertools
pdb="1ubq.pdb"

begin_time = time.time()


u = mda.Universe(pdb,pdb)
ca_atoms = u.select_atoms("name CA")
raw_positions = ca_atoms.positions
print(f"There are {len(raw_positions)} residues in {pdb}")


ave_positions = 0.5*(raw_positions[:-1,:]+raw_positions[1:,:])
bond_vectors = - (raw_positions[:-1,:] - raw_positions[1:,:])
N = len(ave_positions)
nAtom=len(raw_positions)

pair_array=np.asarray(list(itertools.product(ave_positions, ave_positions)))
R_row_idx = pair_array[:,0,:]
R_col_idx = pair_array[:,1,:]
R_diff = R_row_idx-R_col_idx
R_diff_3d = R_diff.reshape(N,N,3)


# [R(i)-R(j)]/norm[R(i)-R(j)]**3
pair_array=np.asarray(list(itertools.product(ave_positions, ave_positions)))
R_row_idx = pair_array[:,0,:]
R_col_idx = pair_array[:,1,:]
R_diff = (R_row_idx-R_col_idx)
R_diff_3d = R_diff.reshape(N,N,3)


# cross produc term
pair_array=np.asarray(list(itertools.product(bond_vectors, bond_vectors)))
dR_row_idx = pair_array[:,0,:]
dR_col_idx = pair_array[:,1,:]
dR_cross = np.cross(dR_row_idx,dR_col_idx)
dR_cross_3d = dR_cross.reshape(N,N,3)

pair_array=np.asarray(list(itertools.product(raw_positions, raw_positions)))

Distance_row_idx = pair_array[:,0,:]
Distance_col_idx = pair_array[:,1,:]
Distance_diff = Distance_row_idx - Distance_col_idx
Distance_pair = np.linalg.norm(Distance_diff, axis=1).reshape(nAtom,nAtom)

final_G = 0
IDX_i1 = 0
IDX_i2 = 0
IDX_j1 = 0
IDX_j2 = 0
# i1,i2 loop to find native contact, distance < 9
for i1 in range(N):
    for i2 in range(i1+3,N):
        if Distance_pair[i1,i2] < 9.0:
#             print(f"contact:{i1}-{i2}: {Distance_pair[i1,i2]}")

            # Gc_ij
            for j1 in range(i2+1,i2+2):
                for j2 in range(N,N+1):
#             for j1 in range(i2+1,N):
#                 for j2 in range(j1+1,N):
                    Gc_ij = 0.0
                    for i in range(i1,i2):
                        for j in range(j1,j2):
                            Gc_ij += np.dot(R_diff_3d[i,j,:]/np.linalg.norm(R_diff_3d[i,j,:])**3, dR_cross_3d[i,j,:])
                    Gc_ij = Gc_ij/(4*np.pi)
#                     print(f'Gc_ij:{Gc_ij}')
                    if final_G < np.abs(Gc_ij):
                        final_G = np.abs(Gc_ij)
                        IDX_i1 = i1
                        IDX_i2 = i2
                        IDX_j1 = j1
                        IDX_j2 = j2
                        print(f'C-term: {final_G : .2f} {IDX_i1} {IDX_i2} {IDX_j1} {IDX_j2}')
            # Gn_ij
            for j1 in range(0,1):
                for j2 in range(i1-1,i1):

#             for j1 in range(i1):
#                 for j2 in range(j1+1,i1):
                    Gn_ij = 0.0
                    for i in range(i1,i2):
                        for j in range(j1,j2):
                            Gn_ij += np.dot(R_diff_3d[i,j,:]/np.linalg.norm(R_diff_3d[i,j,:])**3, dR_cross_3d[i,j,:])
                    Gn_ij = Gn_ij/(4*np.pi)
#                     print(f'Gn_ij:{Gn_ij}')
                    if final_G < np.abs(Gn_ij):
                        final_G = np.abs(Gn_ij)
                        IDX_i1 = i1
                        IDX_i2 = i2
                        IDX_j1 = j1
                        IDX_j2 = j2
                        print(f'N-term: {final_G : .2f} {IDX_i1} {IDX_i2} {IDX_j1} {IDX_j2}')


end_time = time.time()
total_run_time = end_time - begin_time
print(f'Total execution time: {total_run_time / 60.0:.3f} mins')