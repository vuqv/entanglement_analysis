* To use code: python script_name_here pdb_code_here_without_dot_pdb

for example: python v4_pre_compute_numba.py 1ubq.pdb

* main program: 
   i2-i1>=10 and j2-j1 >=10
  
* v4_precomputed_numba.py: 
    i2-i1 >= 1 and j2-j1 >= 1
  

v4 is more general than v3 but v3 is consistence with reported and faster than v4 (it scans smaller sets) 

Notes about |G|c values:
G>=1: entanglement exist (see original paper for details)

For single frame (pdb file), use single_frame.py for short and more detailed

For trajectory, run caller.py, which will automatically divided trajectory in multiple parts
and call ent_calculations for each part (multiple frames)
./run_caller.sh is perfect for short, but you need to modify the input params.

* For Julia version:
At the moment it can only work with PDB of single model
julia gauss_linking.jl PDBFILE
