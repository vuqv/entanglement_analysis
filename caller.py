import MDAnalysis as mda 
import argparse, os
from subprocess import Popen

parser = argparse.ArgumentParser(description="test arg")
parser.add_argument('-top', '-p', type=str, help='Topology')
parser.add_argument('-traj', '-f', type=str, help='Trajectory')
args = parser.parse_args()

u = mda.Universe(args.top, args.traj)
ca_atoms = u.select_atoms("name CA")
n_blocks=2
n_frames= u.trajectory.n_frames


print(n_frames)
n_frames_per_block = n_frames//n_blocks

blocks = [range(i * n_frames_per_block, (i + 1) * n_frames_per_block) for i in range(n_blocks-1)]
blocks.append(range((n_blocks - 1) * n_frames_per_block, n_frames)) #remaining frames
print(blocks)


# python caller.py -p traj/ash1.pdb -f traj/ash1_subtrajectory_500_frames.xtc

for block in blocks:
    cmd= ['python', 'v3_pre_compute_numba.py', '-p', args.top,
     '-f', args.traj,'-b', str(block.start),'-e',str(block.stop)]
    Popen(cmd)