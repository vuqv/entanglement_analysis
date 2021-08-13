import MDAnalysis as mda
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

from multiprocessing import cpu_count
n_jobs = cpu_count()

u = mda.Universe('ash1.pdb', 'ash1_subtrajectory_500_frames.xtc')
protein = u.select_atoms('protein')

def radgyr(atomgroup, masses, total_mass=None):
    # coordinates change for each frame
    coordinates = atomgroup.positions
    center_of_mass = atomgroup.center_of_mass()

    # get squared distance from center
    ri_sq = (coordinates-center_of_mass)**2
    # sum the unweighted positions
    sq = np.sum(ri_sq, axis=1)
    sq_x = np.sum(ri_sq[:,[1,2]], axis=1) # sum over y and z
    sq_y = np.sum(ri_sq[:,[0,2]], axis=1) # sum over x and z
    sq_z = np.sum(ri_sq[:,[0,1]], axis=1) # sum over x and y

    # make into array
    sq_rs = np.array([sq, sq_x, sq_y, sq_z])

    # weight positions
    rog_sq = np.sum(masses*sq_rs, axis=1)/total_mass
    # square root and return
    return np.sqrt(rog_sq)

import dask
import dask.multiprocessing
from dask.distributed import Client

if __name__ == '__main__':
    # dask.freeze_support()
    # client = Client()
    # user code follows
    client = Client(n_workers=n_jobs)
    job_list = []
    for frame in u.trajectory:
        job_list.append(dask.delayed(radgyr(atomgroup=protein,
                                            masses=protein.masses,
                                            total_mass=np.sum(protein.masses))))

    result = dask.compute(job_list)