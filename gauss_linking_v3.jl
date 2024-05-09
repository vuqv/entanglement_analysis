#=
This script modifies the v2 version, which only report the maximal linking number in structure.
This script will report details for every contacts in the structure so that we can compare with whlGLN from Ed's group.
=#
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
        " and topology: ",
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

println("Information about input file:")
println("Number of Residues:", length(resids))
println("Number of frames: ", t.nframe)
# checking number of frames and related stuffs
begin_frame = parsed_args["begin"]
println(begin_frame)
if parsed_args["end"] == 0 || parsed_args["end"] > t.nframe
    end_frame = t.nframe
else
    end_frame = parsed_args["end"]
end
println(end_frame)
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
io = open(filename, "w")
@printf(io, "#   frame \t i1 \t i2 \t j1 \t j2 \t Max(Gc)\n")
# End of output preparation

println("(i1 \t i2): \t (threading N) \t GLN_N \t (threading_C) \t GLN_C")

start_time = time_ns()
for frame = begin_frame:increment_num_frames:end_frame
    # single frame
    coor = reshape(@view(t["protein and atomname CA"].xyz[frame, :]), (3, n_atoms))
    """
        @. indicates operator here is not working on vector.
        slicing array make a copy of sub-array => use @view macro will reduce
            memory allocation and speedup computational time
    """
    Rcoor = @. 0.5 * (@view(coor[:, 1:n_atoms-1]) + @view(coor[:, 2:n_atoms]))
    dRcoor = @. @view(coor[:, 2:n_atoms]) - @view(coor[:, 1:n_atoms-1])

    # len_coor = size(Rcoor)[1]
    R_diff = reshape(
        [
            @. @view(Rcoor[:, i]) - @view(Rcoor[:, j]) for j = 1:len_coor,
            i = 1:len_coor
        ],
        (len_coor, len_coor),
    )
    dR_cross = reshape(
        [
            cross(@view(dRcoor[:, i]), @view(dRcoor[:, j])) for
            j = 1:len_coor, i = 1:len_coor
        ],
        (len_coor, len_coor),
    )
    """ precompute every element of term Gauss double summation.
        Julia is col-oriented programming language. Write programe for col first
        will boot performance.
    """
    dot_cross_matrix = zeros(Float64, (len_coor, len_coor))
    @threads for j = 1:len_coor
        for i = 1:len_coor
            """@inbounds: developers promise all variable are in bounds
                - program and compiler no need to spend time to check.
                - use with cautions.
            @fastmath: tell compiler that only need accuracy in number, not IEEE standard.
            """
            @fastmath @inbounds dot_cross_matrix[i, j] =
                dot((R_diff[i, j] / (norm(R_diff[i, j])^3)), dR_cross[i, j])
        end
    end
    # get contact list in frame
    # We don't know exactly how many contacts in current frame.
    contact_list = []
    # pair_dis = zeros(Float64, n_atoms, n_atoms)
    pair_dis = pairwise(euclidean, coor, dims = 2)
    for i1 = 1:n_atoms-10, i2 = i1+10:n_atoms
        if pair_dis[i1, i2] <= 9.0
            push!(contact_list, (i1, i2))
        end
    end

    """
    This trick increases distance between two variable of array in heap memory.
    By default, this is 1.
    in case of multithreading:
        different thread writes to different var in different cache line -
        -thread does not need to wait for eachother.
    """
    space = 16
    # results = [Tuple{Float64, Int64, Int64, Int64, Int64}[] for _ in 1:nthreads()*space]
    # global results of whole proteins, each thread will has it own value to get max GLN of al nc it working on.
    results = zeros(Float64, (nthreads() * space, 5))
    @threads for cl in contact_list
        i1, i2 = cl
        # GN
        GLN_N, loop_i1, loop_i2, thread_N_j1, thread_N_j2 = 0.0, i1, i2, 0, 0
        for j1 = 1:i1-10, j2 = j1+10:i1-1
            @fastmath @inbounds res =
                sum(@view(dot_cross_matrix[i1:i2-1, j1:j2-1])) / (4 * pi)
            if abs(res) >= abs(GLN_N)
                GLN_N, thread_N_j1, thread_N_j2 = res, j1, j2
#                 println("GLN N terminal")
#                 @printf("%d \t %d \t %d \t %d \t %.3f\n", loop_i1, loop_i2, thread_N_j1, thread_N_j2, GLN_N)
            end
        end
#         println("Final result for N terminal")
#         @printf("%d \t %d \t %d \t %d \t %.3f\n", loop_i1, loop_i2, thread_N_j1, thread_N_j2, GLN_N)
        # GC
        GLN_C, loop_i1, loop_i2, thread_C_j1, thread_C_j2 = 0.0, i1, i2, 0, 0
        for j1 = i2+1:len_coor-10, j2 = j1+10:len_coor
            @fastmath @inbounds res =
                sum(@view(dot_cross_matrix[i1:i2-1, j1:j2-1])) / (4 * pi)
            if abs(res) >= abs(GLN_C)
                GLN_C, thread_C_j1, thread_C_j2 = res, j1, j2
#                 println("GLN C terminal")
#                 @printf("%d \t %d \t %d \t %d \t %.3f\n", loop_i1, loop_i2, thread_C_j1, thread_C_j2, GLN_C)
            end
        end
#         println("Final result for C terminal")
        @printf("(%3d \t %3d): \t (%3d \t %3d) \t %6.3f \t (%3d \t %3d) \t %6.3f\n", loop_i1, loop_i2, thread_N_j1, thread_N_j2, GLN_N, thread_C_j1, thread_C_j2, GLN_C)
    end
end
close(io)
end_time = time_ns()
dt = (end_time - start_time) / 10^9
println("Execution time: ", dt, "(s)")
