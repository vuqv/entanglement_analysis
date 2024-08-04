import BioStructures as bs
# import MDToolbox as mdt
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
    structure = bs.read(pdb_file, bs.PDBFormat)
    
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

"""
    get_contacts(pdb_file::String, bfactor_threshold::Union{Int64, Float64}=0.0)

This function identifies contacts between residues in a protein structure.

# Arguments
- `pdb_file::String`: Path to the PDB file.
- `bfactor_threshold::Union{Int64, Float64}=0.0`: Threshold for the B-factor (temperature factor) to consider residues in contact.

# Returns
- `contacts::Array{Tuple{Int32, Int32}}`: Sorted array of unique residue pairs that are in contact.

# Details
For the AlphaFold structure, we require that the confidence score of the contact residues is greater than or equal to 70 (confident or high-confidence).
"""
function get_contacts(pdb_file::String, bfactor_threshold::Union{Int64, Float64}=0.0)
    # Read the structure from the PDB file
    structure = bs.read(pdb_file, bs.PDBFormat)

    # Count the number of heavy atoms
    n_atoms = bs.countatoms(structure, bs.heavyatomselector)

    # Collect coordinates of heavy atoms
    coords = [atom.coords for atom in bs.collectatoms(structure, bs.heavyatomselector)]
    
    # Flatten and reshape coordinates into a 3 x n_atoms matrix
    flatten_coords = vcat(coords...)
    mat_coords = reshape(flatten_coords, 3, n_atoms)
    
    # Compute pairwise distances between atoms
    # pairwise and Euclidean methods are from Distance module.
    pairwise_distance = pairwise(Euclidean(), mat_coords; dims=2)
    
    # Collect heavy atoms and their residue indices
    heavyatoms = bs.collectatoms(structure, bs.heavyatomselector)
    resid_atom = [parse(Int32, bs.resid(atom)) for atom in heavyatoms]
    
    # Initialize a set to store unique contacts
    contact_set = Set{Tuple{Int32, Int32}}()

    # # Check for contacts based on distance, residue separation, and B-factor threshold
    for i in 1:n_atoms
        for j in i+1:n_atoms
            if pairwise_distance[i, j] <= 4.5 && abs(resid_atom[j] - resid_atom[i]) >= 10 && min(bs.tempfactor(heavyatoms[i]), bs.tempfactor(heavyatoms[j])) >= bfactor_threshold
                push!(contact_set, (resid_atom[i], resid_atom[j]))
            end
        end
    end

    # Convert the set of contacts to a sorted array
    contacts = sort(collect(contact_set), by = x -> (x[1], x[2]))

    return contacts
end


function wh_GLN(coor::Matrix{Float64}, contacts)
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

    Rcoor = 0.5 .* (@view(coor[:, 1:len_coor]) + @view(coor[:, 2:n_atoms]))
    dRcoor =@view(coor[:, 2:n_atoms]) .- @view(coor[:, 1:len_coor])

    R_diff = [@view(Rcoor[:, i]) .- @view(Rcoor[:, j]) for i in 1:len_coor, j in 1:len_coor]
    dR_cross = [LinearAlgebra.cross(@view(dRcoor[:, i]), @view(dRcoor[:, j])) for i in 1:len_coor, j in 1:len_coor]

    dot_cross_matrix = zeros(Float64, len_coor, len_coor)
    @inbounds @fastmath @threads for j in 1:len_coor
        for i in 1:len_coor
            R_diff_ij = R_diff[i, j]
            dR_cross_ij = dR_cross[i, j]
            dot_cross_matrix[i, j] = LinearAlgebra.dot(R_diff_ij / (norm(R_diff_ij)^3), dR_cross_ij)
        end
    end

    # pair_dis = Distances.pairwise(Distances.Euclidean(), coor; dims = 2)
    # contact_list = [(i1, i2) for i1 in 1+exclusion:n_atoms-10-exclusion for i2 in i1+10:n_atoms-exclusion if pair_dis[i1, i2] <= 9.0]

    space = 16
    results = zeros(Float64, nthreads() * space, 5)
# handle for whGLN
    @inbounds @fastmath @threads for cl in contacts #contact_list
        i1, i2 = cl
        # @printf("%3d \t %3d \n", i1, i2)
        # GN
        if i1 > 12
            """
            Why 12: For N terminal, range of N-thread is from j1=6 and j2 = i1-6
            if i1 < 12 meanning [j1:j2]=0
            """
            j1, j2 = 6, i1-6 # i1-6 is included
            @fastmath @inbounds res = sum(@view(dot_cross_matrix[i1:i2-1, j1:j2])) / (4 * pi)
            GLN_N = res
            # @printf("GLN_N: %.3f", GLN_N)
        else
            #if i1 <= 12 meanning N-terminal is not consider
            GLN_N = 0.0
            # @printf("GLN_N: %.3f", GLN_N)
        end
        # GC
        if i2 < len_coor - 12
            j1, j2 = i2+6, len_coor - 6 #len_coor -6 is included
            @fastmath @inbounds res = sum(@view(dot_cross_matrix[i1:i2-1, j1:j2])) / (4 * pi)
            GLN_C = res
            # @printf("GLN_C: %.3f", GLN_C)
        else
            GLN_C = 0.0
            # @printf("GLN_C: %.3f", GLN_C)
        end

        if abs(GLN_N) >= 0.6 || abs(GLN_C) >=0.6
            @printf("(%3d \t %3d): \t %6.3f \t %6.3f\n", i1, i2, GLN_N, GLN_C)
        end
        # @printf("(%3d \t %3d): \t (%3d \t %3d) \t %6.3f \t (%3d \t %3d) \t %6.3f\n", loop_i1, loop_i2, thread_N_j1, thread_N_j2, GLN_N, thread_C_j1, thread_C_j2, GLN_C)
    end

# maxGLN
    # @inbounds @fastmath @threads for cl in contacts #contact_list
    #     i1, i2 = cl
    #     # check for N-term segment
    #     for j1 in 1:i1-10, j2 in j1+10:i1-1
    #         res = abs(sum(dot_cross_matrix[i1:i2-1, j1:j2-1]) / (4 * pi))
    #         if res >= results[threadid() * space, 1]
    #             results[threadid() * space, :] = [res, i1, i2, j1, j2]
    #         end
    #     end
        
    #     # check for C-term segment
    #     for j1 in i2+1:len_coor-10, j2 in j1+10:len_coor
    #         res = abs(sum(dot_cross_matrix[i1:i2-1, j1:j2-1]) / (4 * pi))
    #         if res >= results[threadid() * space, 1]
    #             results[threadid() * space, :] = [res, i1, i2, j1, j2]
    #         end
    #     end
    # end

    # idx_max_gc = argmax(results[:, 1])
    # max_gc, idx_i1, idx_i2, idx_j1, idx_j2 = results[idx_max_gc, :]
    # return Int(idx_i1), Int(idx_i2), Int(idx_j1), Int(idx_j2), max_gc

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
        "--PDB", "-f"
            help = "PDB Input"
            arg_type = String
        "--Output", "-o"
            help = "Output directory"
            arg_type = String
            default = "./"
        "--BFactor", "-b"
            help =  "B-factor threshold, encodes of confidence score for AF structure"
            arg_type = String
            default = "0.0"
    end

    args = parse_args(s)
    # Convert B-factor threshold to numeric type
    try
        args["BFactor"] = parse(Float64, args["BFactor"])
    catch
        error("Invalid B-factor threshold: $(args["BFactor"]). Must be a number.")
    end
    return args

    # return parse_args(s)
end

function warmup_calculations(pdb_file::String)
    warmup_coor = read_coordinates(pdb_file)
    warmup_contacts = get_contacts(pdb_file)
    GLN(warmup_coor, warmup_contacts)
end

function main()
    args = parse_commandline()
    # pdb_file = args["PDB"]
    pdb_file = get(args, "PDB", "")
    output_dir = get(args, "Output", "./")
    bfactor_threshold = get(args, "BFactor", 0.0)

    if isempty(pdb_file)
        println("Error: PDB file argument is required.")
        return
    end
    
    base_name = split(basename(pdb_file), ".")[1]

    # Create output directory if it doesn't exist
    if !isdir(output_dir)
        mkpath(output_dir)
        println("Created output directory: $output_dir")
    end

    # prepare output file
    # output_filename = joinpath(output_dir, "MAX_GLN_$base_name.dat")
    # io = open(output_filename, "w")

    try
        coor = read_coordinates(pdb_file)
        contacts = get_contacts(pdb_file, bfactor_threshold)
        wh_GLN(coor, contacts)
        # i1, i2, j1, j2, maxG = GLN(coor, contacts)
        # @printf("%s; [(%d, %d) | (%d, %d)]; %.3f\n", base_name, i1, i2, j1, j2, maxG)
        # @printf(io, "%s; [(%d, %d) | (%d, %d)]; %.3f\n", base_name, i1, i2, j1, j2, maxG)
    catch err
        println("Error processing $pdb_file: $err")
    finally
        # close(io)
        println("DONE!!!")
    end

end

start_time = time_ns()
# warmup_calculations("warmup.pdb")
main()
dt = (time_ns() - start_time) / 10^9
println("Execution time: ", dt, "(s)")