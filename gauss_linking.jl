start_time = time_ns()
using MDToolbox, Distances, LinearAlgebra, ArgParse, Printf

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table s begin
#         "--opt1"
#             help = "an option with an argument"
        "--traj", "-f"
            help = "Trajectory"
            arg_type = String
            default = ""
        "--top", "-p"
            help = "Topology"
            arg_type = String
            default = ""

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
nframes = t.nframe
# total_traj = t["atomname CA"].xyz
@printf("frame \t i1 \t i2 \t j1 \t j2 \t Max(Gc)\n")
start_time = time_ns()
for frame in 1:nframes
    # single frame
    coor = reshape(t["atomname CA"].xyz[frame,:], (3,n_atoms))'
    Rcoor = @. 0.5*(coor[1:n_atoms-1,:]+coor[2:n_atoms,:])
    dRcoor = @. coor[2:n_atoms,:]-coor[1:n_atoms-1,:]
    len_coor = size(Rcoor)[1]
    R_diff = reshape([@. Rcoor[i,:]-Rcoor[j,:] for j in 1:len_coor for i in 1:len_coor], (len_coor,len_coor));
    dR_cross = reshape([cross(dRcoor[i,:],dRcoor[j,:]) for j in 1:len_coor for i in 1:len_coor], (len_coor,len_coor));
    dot_cross_matrix = zeros(Float64,(len_coor,len_coor));
    @inbounds @simd for j in 1: len_coor
        for i in 1: len_coor
            dot_cross_matrix[i,j] = dot((R_diff[i,j]/(norm(R_diff[i,j])^3)), dR_cross[i,j])
        end
    end
    # get contact list in frame
    pair_dis = pairwise(euclidean, coor, dims=1)
    contact_list = []
    @simd for i1 in 1:len_coor-10
        for i2 in i1+10:len_coor
            if pair_dis[i1,i2] <= 9.0
                push!(contact_list, (i1,i2))
            end
        end
    end
    

    max_gc, idx_i1,idx_i2,idx_j1,idx_j2 = 0.0, 0, 0, 0, 0
    @inbounds @simd for cl in contact_list
        i1,i2 = cl
        for j1 in 1:i1-10
            # res = 0.0
            @inbounds @simd for j2 in j1+10:i1-1
                @fastmath res = abs(sum(dot_cross_matrix[i1:i2-1,j1:j2-1])/(4*pi))
                if res >= max_gc
                    max_gc, idx_i1,idx_i2,idx_j1,idx_j2  = res, i1, i2, j1, j2
                end
            end
        end

        @inbounds @simd for j1 in i2+1:len_coor-10
            @inbounds @simd for j2 in j1+10:len_coor
                @fastmath res = abs(sum(dot_cross_matrix[i1:i2-1,j1:j2-1])/(4*pi))
                if res >= max_gc
                    max_gc, idx_i1,idx_i2,idx_j1,idx_j2  = res, i1, i2, j1, j2
                end
            end
        end
    end

#     max_gc_idx = argmax(results[:,5])
#     println("frame: ", frame, ", MaxGc: ", idx_i1,idx_i2,idx_j1,idx_j2, max_gc)
    @printf("%d \t %d \t %d \t %d \t %d \t %.3f\n", frame, idx_i1, idx_i2, idx_j1, idx_j2, max_gc)
end

end_time = time_ns()
dt = (end_time-start_time)/10^9
println("Execution time: ", dt,  "(s)")