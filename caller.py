"""
script to split into multiple calculation.
subprocess.Popen(): call threads without waiting while run() will wait for output before run next thread
here, we may consider to split trajectory into multiple part because storage is larger than memory
using mdconvert to split
"""
import argparse
from subprocess import Popen
import subprocess
import os
from multiprocessing import cpu_count
import MDAnalysis as mda

parser = argparse.ArgumentParser(description="test arg")
parser.add_argument('-top', '-p', type=str, help='Topology')
parser.add_argument('-traj', '-f', type=str, help='Trajectory')
parser.add_argument('-nthreads', '-nt', type=int, help='number of threads', default=cpu_count())
parser.add_argument('-output', '-o', type=str, help='output file', default='final_res.dat')
args = parser.parse_args()

u = mda.Universe(args.top, args.traj)
ca_atoms = u.select_atoms("name CA")
n_blocks = args.nthreads
n_frames = u.trajectory.n_frames

# print(n_frames)
n_frames_per_block = n_frames // n_blocks

blocks = [range(i * n_frames_per_block, (i + 1) * n_frames_per_block) for i in range(n_blocks - 1)]
blocks.append(range((n_blocks - 1) * n_frames_per_block, n_frames))  # remaining frames

if os.path.exists('temp'):
    pass 
else:
    os.makedirs('temp')

# print(blocks)
jobs = []
jobs_id = []
for block in blocks:
    cmd = ['python', 'ent_calculation.py', '-p', args.top,
           '-f', args.traj, '-b', str(block.start), '-e', str(block.stop)]
    job = Popen(cmd)
    jobs.append(job)
    
# wait until all jobs finish
for job in jobs:
    job.wait()
    
# combine results
if os.path.exists(f'{args.output}'):
    os.system(f'rm -f {args.output}')
for block in blocks:
    os.system(f'grep -v Total temp/res_{block.start}_{block.stop} >> {args.output}')
