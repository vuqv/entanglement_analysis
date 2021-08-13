import argparse
from subprocess import Popen

import MDAnalysis as mda

parser = argparse.ArgumentParser(description="test arg")
parser.add_argument('-top', '-p', type=str, help='Topology')
parser.add_argument('-traj', '-f', type=str, help='Trajectory')
args = parser.parse_args()

u = mda.Universe(args.top, args.traj)
ca_atoms = u.select_atoms("name CA")
n_blocks = 2
n_frames = u.trajectory.n_frames

print(n_frames)
n_frames_per_block = n_frames // n_blocks

blocks = [range(i * n_frames_per_block, (i + 1) * n_frames_per_block) for i in range(n_blocks - 1)]
blocks.append(range((n_blocks - 1) * n_frames_per_block, n_frames))  # remaining frames
print(blocks)

for block in blocks:
    cmd = ['python', 'ent_calculation.py', '-p', args.top,
           '-f', args.traj, '-b', str(block.start), '-e', str(block.stop)]
    Popen(cmd)
