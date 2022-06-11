* To use code: 
    ```python gauss_linking.py -f TRAJ -p TOP -b BEGIN -e END -skip SKIP_FRAME -nproc NUMTHREAD```


Notes about |G|c values:
G>=1: entanglement exist (see original paper for details)

For single frame (pdb file), use single_frame.py for short and more detailed

For trajectory, run caller.py, which will automatically divided trajectory in multiple parts
and call ent_calculations for each part (multiple frames)
./run_caller.sh is perfect for short, but you need to modify the input params.

* For Julia version:
Full command:  
    ```julia -J SYS_IMAGE.so -t NUM_THREADS gauss_linking.jl -f TRAJ -p TOP -b START_FRAME -e END_FRAME -s SKIP_FRAME ```
simple command:
    ```julia gauss_linking.jl -f PDBFILE```

