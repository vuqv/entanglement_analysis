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

        "--exclusion", "-x"
        help = "exclusion of terminal"
        arg_type = Int64
        default = 0

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

        "--out_dir", "-o"
        help = "skip every frame"
        arg_type = String
        default = "./"
    end
    return parse_args(s)
end
parsed_args = parse_commandline()

# working with PDB contains multiple models
if parsed_args["top"] != ""
    t = mdload(parsed_args["top"])
    t = mdload(parsed_args["traj"], top = t)
    println(
        "Working on trajectory: ",
        parsed_args["traj"],
        "and topology: ",
        parsed_args["top"],
    )
else
    t = mdload(parsed_args["traj"])
    println("Working on: ", parsed_args["traj"])
end


end_time = time_ns()
println("Initialize time: ", (end_time - start_time) / 10^9, "(s)")

const n_atoms = t["protein and atomname CA"].natom
const len_coor = n_atoms - 1 #length of average coordinate
resids = t["atomname CA"].resid
# checking number of frames and related stuffs
begin_frame = parsed_args["begin"]

if parsed_args["end"] == 0 || parsed_args["end"] > t.nframe
    end_frame = t.nframe
else
    end_frame = parsed_args["end"]
end

# parse exclusion of tails
exclusion = parsed_args["exclusion"]

increment_num_frames = parsed_args["skip"]
# prepare file for output
if parsed_args["out_dir"] == "./"
    out_dir = parsed_args["out_dir"]
    println("Output directory is not specified. Use current folder as default.")
else
    # if output directory is specified, check if it exists
    if isdir(parsed_args["out_dir"])
        # do something if dir exists
        println("Output directory is specified and existed.")
        out_dir = parsed_args["out_dir"] * "/"
    else
        println(
            "Output directory is specified and does not existed. Making folder for results",
        )
        mkdir(parsed_args["out_dir"])
        out_dir = parsed_args["out_dir"] * "/"
    end
end

filename =
    out_dir * split(parsed_args["traj"], ('.', '/'))[end - 1] * "_results.txt"
# io = open(filename, "w")
# @printf(io, "#   frame \t i1 \t i2 \t j1 \t j2 \t Max(Gc)\n")
# End of output preparation

println("frame \t i1 \t i2 \t j1 \t j2 \t Max(Gc)")

# Turn main calculation to function

function GLN(coor)
    n_atoms = size(coor, 2)
    exclusion = 1  # Assuming exclusion is defined somewhere
    len_coor = n_atoms - 1

    Rcoor = 0.5 .* (coor[:, 1:len_coor] + coor[:, 2:n_atoms])
    dRcoor = coor[:, 2:n_atoms] .- coor[:, 1:len_coor]

    R_diff = [Rcoor[:, i] .- Rcoor[:, j] for i in 1:len_coor, j in 1:len_coor]
    dR_cross = [cross(dRcoor[:, i], dRcoor[:, j]) for i in 1:len_coor, j in 1:len_coor]

    dot_cross_matrix = zeros(Float64, len_coor, len_coor)
    @inbounds @fastmath @threads for j in 1:len_coor
        for i in 1:len_coor
            R_diff_ij = R_diff[i, j]
            dR_cross_ij = dR_cross[i, j]
            dot_cross_matrix[i, j] = dot(R_diff_ij / (norm(R_diff_ij)^3), dR_cross_ij)
        end
    end

    pair_dis = pairwise(Euclidean(), coor; dims = 2)
    contact_list = [(i1, i2) for i1 in 1+exclusion:n_atoms-10-exclusion for i2 in i1+10:n_atoms-exclusion if pair_dis[i1, i2] <= 9.0]

    space = 16
    results = zeros(Float64, nthreads() * space, 5)

    @inbounds @fastmath @threads for cl in contact_list
        i1, i2 = cl
        for j1 in 1:i1-10, j2 in j1+10:i1-1
            res = abs(sum(dot_cross_matrix[i1:i2-1, j1:j2-1]) / (4 * π))
            if res >= results[threadid() * space, 1]
                results[threadid() * space, :] = [res, i1, i2, j1, j2]
            end
        end

        for j1 in i2+1:len_coor-10, j2 in j1+10:len_coor
            res = abs(sum(dot_cross_matrix[i1:i2-1, j1:j2-1]) / (4 * π))
            if res >= results[threadid() * space, 1]
                results[threadid() * space, :] = [res, i1, i2, j1, j2]
            end
        end
    end

    idx_max_gc = argmax(results[:, 1])
    max_gc, idx_i1, idx_i2, idx_j1, idx_j2 = results[idx_max_gc, :]

    if idx_i1 == 0
        @printf("%d \t %d \t %d \t %d \t %.3f\n", 0, 0, 0, 0, 0)
    else
        @printf("%d \t %d \t %d \t %d \t %.3f\n", Int(idx_i1), Int(idx_i2), Int(idx_j1), Int(idx_j2), max_gc)
    end
end


start_time = time_ns()
for frame = begin_frame:increment_num_frames:end_frame
    # single frame
    coor = reshape(@view(t["protein and atomname CA"].xyz[frame, :]), (3, n_atoms))
    """
        @. indicates operator here is not working on vector.
        slicing array make a copy of sub-array => use @view macro will reduce
            memory allocation and speedup computational time
    """
    GLN(coor)

end
# close(io)
end_time = time_ns()
dt = (end_time - start_time) / 10^9
println("Execution time: ", dt, "(s)")
