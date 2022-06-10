using BioStructures
using Distances
using LinearAlgebra
using Printf

start_time = time_ns()

struc = read("1ubq.pdb", PDB)
# select all CA atoms
calphas = collectatoms(struc, calphaselector);
n_atoms = length(calphas)
# initialize coordinate matrix
coor = zeros(n_atoms, 3)
for i in eachindex(calphas)
    coor[i,:] = coords(calphas[i])
end

# average coordinate
Rcoor = 0.5*(coor[1:n_atoms-1,:]+coor[2:n_atoms,:]);
# bond vector
dRcoor = coor[2:n_atoms,:]-coor[1:n_atoms-1,:];

# pairwise distance between two CA atoms.
# pair_dis= pairwise(euclidean, Rcoor, dims=1);
pair_dis2 = pairwise(euclidean, coor, dims=1)
# generate contact list
len_coor = size(Rcoor)[1]
contact_list = []

for i1 in 1:len_coor-10
    for i2 in i1+10:len_coor
        if pair_dis2[i1,i2]<=9
            push!(contact_list, (i1,i2))
        end
    end
end

function cal_gln(i1,i2,j1,j2)
    gln_ij = 0
    for i in i1:i2-1
        for j in j1:j2-1
            gln_ij += dot((Rcoor[i,:]-Rcoor[j,:])/(norm(Rcoor[i,:]-Rcoor[j,:]))^3,cross(dRcoor[i,:],dRcoor[j,:]))
        end
    end
    return gln_ij/(4*pi)
end

results = []
for cl in contact_list
    i1,i2 = cl
    for j1 in 1:i1-10
        for j2 in j1+10:i1-1
            res = @time cal_gln(i1,i2,j1,j2)
            push!(results, abs(res))
        end
    end
    
    for j1 in i2+1:len_coor-10
        for j2 in j1+10:len_coor
            res = @time cal_gln(i1,i2,j1,j2)
            push!(results, abs(res))
        end
    end
    
end

println(maximum(results))
end_time = time_ns()

dt = (end_time-start_time)/10^9
println("Execution time: (s)", dt)