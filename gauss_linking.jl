start_time = time_ns()
using MDToolbox, Distances, LinearAlgebra, ArgParse, Printf, Base.Threads

# Parse argument
function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
        "--traj", "-f"
            help = "Trajectory"
            arg_type = String
            default = ""
        "--top", "-p"
            help = "Topology"
            arg_type = String
            default = ""
        "--begin", "-b"
            help = "Starting Frame"
            arg_type = Int64
            default = 1
        "--end", "-e"
            help = "End Frame"
            arg_type = Int64
            default = 0
        "--skip", "-s"
            help = "skip every frame"
            arg_type = Int64
            default = 1
    end
    return parse_args(s)
end
parsed_args = parse_commandline()

# working with PDB contains multiple models
if parsed_args["top"] != ""
    t = mdload(parsed_args["top"]);
    t = mdload(parsed_args["traj"], top = t)
    println("Working on trajectory: ", parsed_args["traj"], "and topology: ", parsed_args["top"])
else
    t = mdload(parsed_args["traj"])
    println("Working on: ", parsed_args["traj"])
end
  

end_time = time_ns()
println("Initialize time: ", (end_time-start_time)/10^9, "(s)")

n_atoms = t["atomname CA"].natom
resids = t["atomname CA"].resid
# checking number of frames and related stuffs
if parsed_args["end"] == 0
    nframes = t.nframe
elseif parsed_args["end"] > t.nframe
    nframes = t.nframe
else
    nframes = parsed_args["end"]
end

increment_num_frames = parsed_args["skip"]

println("frame \t i1 \t i2 \t j1 \t j2 \t Max(Gc)")
start_time = time_ns()
for frame in 1:increment_num_frames:nframes
    # single frame
    coor = reshape(t["atomname CA"].xyz[frame,:], (3,n_atoms))'
    # @. indicates operator here is not working on vector.
    Rcoor = @. 0.5*(coor[1:n_atoms-1,:]+coor[2:n_atoms,:])
    dRcoor = @. coor[2:n_atoms,:]-coor[1:n_atoms-1,:]
    
    len_coor = size(Rcoor)[1]
    R_diff = reshape([@. Rcoor[i,:]-Rcoor[j,:] for j in 1:len_coor for i in 1:len_coor], (len_coor,len_coor));
    dR_cross = reshape([cross(dRcoor[i,:],dRcoor[j,:]) for j in 1:len_coor for i in 1:len_coor], (len_coor,len_coor));
    # precompute every element of term Gauss double summation
    dot_cross_matrix = zeros(Float64,(len_coor,len_coor));
    
    @threads for i in 1:len_coor
        for j in 1: len_coor
            # @inbounds: developers promise all variable are in bounds- program and compiler no need to spend time to check
            @fastmath @inbounds dot_cross_matrix[i,j] = dot((R_diff[i,j]/(norm(R_diff[i,j])^3)), dR_cross[i,j])
        end
    end
    # get contact list in frame
    contact_list = []
    pair_dis = pairwise(euclidean, coor, dims=1)
    for i1 in 1:n_atoms-10, i2 in i1+10:n_atoms
            if pair_dis[i1,i2] <= 9.0
                push!(contact_list, (i1,i2))
            end
    end

    """
    this increase distance between two variable of array in heap memory, by default, this is 1.
    in case of multithreading: 
        different threads write to different var in different cache line then thread does not need to wait for eachother.
    """
    space = 16 
    # results = [Tuple{Float64, Int64, Int64, Int64, Int64}[] for _ in 1:nthreads()*space]
    results = zeros(Float64, (nthreads()*space, 5))
    @threads for cl in contact_list
        i1,i2 = cl
        for j1 in 1:i1-10, j2 in j1+10:i1-1
            @fastmath @inbounds res = abs(sum(dot_cross_matrix[i1:i2-1,j1:j2-1])/(4*pi))
                if res >= results[threadid()*space,1]
                    results[threadid()*space,:]  = [res, i1, i2, j1, j2]
                end
        end

        for j1 in i2+1:len_coor-10, j2 in j1+10:len_coor
            @fastmath @inbounds res = abs(sum(dot_cross_matrix[i1:i2-1,j1:j2-1])/(4*pi))
                if res >= results[threadid()*space,1]
                    results[threadid()*space,:]  = [res, i1, i2, j1, j2]
                end
        end
    end
    idx_max_gc = argmax(results[:,1])
    max_gc, idx_i1,idx_i2,idx_j1,idx_j2  = results[idx_max_gc, 1], results[idx_max_gc, 2], results[idx_max_gc, 3], results[idx_max_gc, 4], results[idx_max_gc, 5]
    @printf("%d \t %d \t %d \t %d \t %d \t %.3f\n", frame, resids[Int64(idx_i1)], resids[Int64(idx_i2)], resids[Int64(idx_j1)], resids[Int64(idx_j2)], max_gc)
    # @printf("%d \t %d \t %d \t %d \t %d \t %.3f\n", frame, idx_i1, idx_i2, idx_j1, idx_j2, max_gc)
end

end_time = time_ns()
dt = (end_time-start_time)/10^9
println("Execution time: ", dt,  "(s)")
