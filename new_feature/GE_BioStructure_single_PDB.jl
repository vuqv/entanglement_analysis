import BioStructures as bs
# import MDToolbox as mdt
# using HDF5
using Distances, LinearAlgebra, ArgParse, Printf, Base.Threads
# import BenchmarkTools as bmt

function read_coordinates(pdb_file::String) :: Matrix{Float64}
    """
    read_coordinates(pdb_file::String) -> Matrix{Float64}

    Read the coordinates of all CA (alpha carbon) atoms from a PDB file.
    
    # Arguments
    - `pdb_file::String`: The path to the PDB file.
    
    # Returns
    - `Matrix{Float64}`: A 3xN matrix where N is the number of CA atoms, and each column represents the coordinates of a CA atom.
    """

    
    # load PDB file
    structure = bs.read(pdb_file, bs.PDB)
    
    # Number of CA atoms-TODO: select CA of protein only
    n_residues = bs.countatoms(structure, bs.calphaselector)
    
    # get coordinate of all CA atoms
    coords = [atom.coords for atom in bs.collectatoms(structure, bs.calphaselector)]
    
    # Processing coordinate
    # Flatten the coordinates
    flatten_coords = vcat(coords...)
    
    # Reshape to 3xN matrix
    mat_coords = reshape(flatten_coords, 3, n_residues)
    
    return mat_coords
end

function read_coordinates_multiple_files(pdb_files::Vector{String}) :: Vector{Matrix{Float64}}
    """
    read_coordinates_multiple_files(pdb_files::Vector{String}) -> Vector{Matrix{Float64}}

    Read the coordinates of all CA atoms from multiple PDB files.
    
    # Arguments
    - `pdb_files::Vector{String}`: A vector of paths to the PDB files.
    
    # Returns
    - `Vector{Matrix{Float64}}`: A vector of 3xN matrices, each representing the coordinates from a PDB file.
    """
    
    all_coords = [read_coordinates(pdb_file) for pdb_file in pdb_files]
    return all_coords
end

function GLN(coor::Matrix{Float64})
    """
    GLN(coor)

    Calculate the Gaussian Linking Number (GLN) for a set of 3D coordinates representing
    a molecular structure.

    The Gaussian Linking Number measures the linking of two closed curves in space.

    # Arguments
    - `coor::Matrix{Float64}`: A 3xN matrix where each column represents the coordinates of 
    a single atom or point in 3D space.

    # Returns
    - `Int, Int, Int, Int, Float64`: Returns a tuple containing:
    - `i1::Int, i2::Int`: Indices of the first curve segment.
    - `j1::Int, j2::Int`: Indices of the second curve segment.
    - `maxG::Float64`: Maximum Gaussian Linking Number value computed.

    # Description
    The function computes the GLN based on the provided coordinates. It calculates distances,
    cross products, and dot products to determine the maximum linking number between segments
    of the molecular structure.

    # Notes
    - This function assumes that `coor` contains coordinates for a closed curve, typically
    representing a molecular chain or a DNA strand.
    - The exclusion and space parameters influence the computation to avoid self-crossing 
    and to optimize performance in multi-threaded environments.
    """
    n_atoms = size(coor, 2)
    exclusion = 0  # Assuming exclusion is defined somewhere
    len_coor = n_atoms - 1

    Rcoor = 0.5 .* (coor[:, 1:len_coor] + coor[:, 2:n_atoms])
    dRcoor = coor[:, 2:n_atoms] .- coor[:, 1:len_coor]

    R_diff = [Rcoor[:, i] .- Rcoor[:, j] for i in 1:len_coor, j in 1:len_coor]
    dR_cross = [LinearAlgebra.cross(dRcoor[:, i], dRcoor[:, j]) for i in 1:len_coor, j in 1:len_coor]

    dot_cross_matrix = zeros(Float64, len_coor, len_coor)
    @inbounds @fastmath @threads for j in 1:len_coor
        for i in 1:len_coor
            R_diff_ij = R_diff[i, j]
            dR_cross_ij = dR_cross[i, j]
            dot_cross_matrix[i, j] = LinearAlgebra.dot(R_diff_ij / (norm(R_diff_ij)^3), dR_cross_ij)
        end
    end

    pair_dis = Distances.pairwise(Distances.Euclidean(), coor; dims = 2)
    contact_list = [(i1, i2) for i1 in 1+exclusion:n_atoms-10-exclusion for i2 in i1+10:n_atoms-exclusion if pair_dis[i1, i2] <= 9.0]

    space = 16
    results = zeros(Float64, nthreads() * space, 5)

    @inbounds @fastmath @threads for cl in contact_list
        i1, i2 = cl
        for j1 in 1:i1-10, j2 in j1+10:i1-1
            res = abs(sum(dot_cross_matrix[i1:i2-1, j1:j2-1]) / (4 * pi))
            if res >= results[threadid() * space, 1]
                results[threadid() * space, :] = [res, i1, i2, j1, j2]
            end
        end

        for j1 in i2+1:len_coor-10, j2 in j1+10:len_coor
            res = abs(sum(dot_cross_matrix[i1:i2-1, j1:j2-1]) / (4 * pi))
            if res >= results[threadid() * space, 1]
                results[threadid() * space, :] = [res, i1, i2, j1, j2]
            end
        end
    end

    idx_max_gc = argmax(results[:, 1])
    max_gc, idx_i1, idx_i2, idx_j1, idx_j2 = results[idx_max_gc, :]
    return Int(idx_i1), Int(idx_i2), Int(idx_j1), Int(idx_j2), max_gc

end

function read_list_pdbs(filename::String) :: Vector{String}
    """

    read_list_pdbs(filename::String) -> Vector{String}

    Reads a file containing indices and PDB paths separated by commas and returns a vector
    of PDB paths.

    # Arguments
    - `filename::String`: The path to the file to be read. The file should contain lines in the format `index, pdb_path`.

    # Returns
    - `Vector{String}`: An array where each element is a PDB path (as a string).


    """
    pdb_files = Vector{String}()
    lines = readlines(filename)
    for line in lines
        parts = split(line, ",")
        if length(parts) >= 2
            idx = strip(parts[1])
            pdb_path = strip(parts[2])
            # println("Index: $idx, PDB Path: $pdb_path")
            push!(pdb_files, pdb_path)
        else
            println("skipping malformed line: $line")
        end
    end
    return pdb_files
end



function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--index", "-i"
            help = "Frame Index"
            arg_type = Int
            default = 1
    end

    return parse_args(s)
end

function main()
    args = parse_commandline()
    frame_idx = args["index"]

    pdb_files = read_list_pdbs("human_all_AF.csv")
    base_path = "/storage/group/epo2/default/for_Viraj/Entanglement_project/Ecoli_Yeast_Human_alpha_folds_ent/Human/Before_TER_PDBs"

    output_filename = string("output/MAX_GC_", frame_idx,".dat")
    io = open(output_filename, "w")


    # Your main code here, using start_idx and end_idx

    # Read list of PDB files need to calculate

    
    file = string(base_path, "/", pdb_files[frame_idx], "_v4.pdb")
    coor = read_coordinates(file)
    i1, i2, j1, j2, maxG = GLN(coor)
    @printf("%d; %s; [(%d, %d) | (%d, %d)]; %.3f\n", frame_idx, pdb_files[frame_idx], i1, i2, j1, j2, maxG)
    @printf(io, "%d; %s; [(%d, %d) | (%d, %d)]; %.3f\n", frame_idx, pdb_files[frame_idx], i1, i2, j1, j2, maxG)
    flush(io)




    close(io)

end

main()
