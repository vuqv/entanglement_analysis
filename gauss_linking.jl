using BioStructures, Distances
# using Printf
using LinearAlgebra
using ArgParse

# s = ArgParseSettings()
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "PDBFile"
            help = "a positional argument"
            required = true
    end

    return parse_args(s)
end
parsed_args = parse_commandline()
# for (arg, val) in parsed_args
#     println("$arg => $val")
#     println(parsed_args["PDBFile"])
# end
println("Working on: ", parsed_args["PDBFile"])
struc = read(parsed_args["PDBFile"], PDB)
# select all CA atoms
calphas = collectatoms(struc, calphaselector);
n_atoms = length(calphas)
# initialize coordinate matrix

start_time = time_ns()

coor = zeros(Float64, n_atoms, 3)
@simd for i in eachindex(calphas)
    coor[i, :] = coords(calphas[i])
end

# average coordinate
Rcoor = @. 0.5*(coor[1:n_atoms-1,:]+coor[2:n_atoms,:]);
# bond vector
dRcoor = @. coor[2:n_atoms,:]-coor[1:n_atoms-1,:];

const len_coor = size(Rcoor)[1]
R_diff = reshape([@. Rcoor[i,:]-Rcoor[j,:] for j in 1:len_coor for i in 1:len_coor], (len_coor,len_coor));
dR_cross = reshape([cross(dRcoor[i,:],dRcoor[j,:]) for j in 1:len_coor for i in 1:len_coor], (len_coor,len_coor));
dot_cross_matrix = zeros(Float64,(len_coor,len_coor));
@inbounds @simd for i in 1: len_coor
    for j in 1: len_coor
        dot_cross_matrix[i,j] = dot((R_diff[i,j]/(norm(R_diff[i,j])^3)), dR_cross[i,j])
    end
end
# pairwise distance between two CA atoms.
pair_dis = pairwise(euclidean, coor, dims=1)

# generate contact list
contact_list = []
@simd for i1 in 1:len_coor-10
    for i2 in i1+10:len_coor
        if pair_dis[i1,i2] <= 9.0
            push!(contact_list, (i1,i2))
        end
    end
end

# Define function to calculate GLN
function cal_gln(i1::Int64,i2::Int64,j1::Int64,j2::Int64)
    gln_ij = 0.0
    @inbounds @simd for i in i1:i2-1
        @inbounds @simd for j in j1:j2-1
            @fastmath gln_ij += dot_cross_matrix[i,j]
        end
    end
    # using * instead of /
    return 0.0795775*gln_ij
    # return gln_ij/(4.0*pi)
end

results = Float64[]
@inbounds @simd for cl in contact_list
    i1,i2 = cl
    for j1 in 1:i1-10
        # res = 0.0
        @inbounds @simd for j2 in j1+10:i1-1
            @fastmath res = cal_gln(i1, i2, j1, j2)
#             @printf("%d %d %d %d %f\n",i1,i2,j1,j2,res)
            push!(results, abs(res))
        end
    end
    
    @inbounds @simd for j1 in i2+1:len_coor-10
        @inbounds @simd for j2 in j1+10:len_coor
            @fastmath res = cal_gln(i1,i2,j1,j2)
#             @printf("%d %d %d %d %f\n",i1,i2,j1,j2,res)
            push!(results, abs(res))
        end
    end
    
end

println(maximum(results))

end_time = time_ns()
dt = (end_time-start_time)/10^9
println("Execution time: ", dt,  "(s)")
