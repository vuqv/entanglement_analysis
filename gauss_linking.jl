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
        "and topology: ",
        parsed_args["top"],
    )
else
    t = mdload(parsed_args["traj"])
    println("Working on: ", parsed_args["traj"])
end


end_time = time_ns()
println("Initialize time: ", (end_time - start_time) / 10^9, "(s)")

const n_atoms = t["atomname CA"].natom
const len_coor = n_atoms - 1 #length of average coordinate
resids = t["atomname CA"].resid
# checking number of frames and related stuffs
begin_frame = parsed_args["begin"]

if parsed_args["end"] == 0 || parsed_args["end"] > t.nframe
    end_frame = t.nframe
else
    end_frame = parsed_args["end"]
end

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
    out_dir * split(parsed_args["traj"], ('.', '/'))[end-1] * "_results.txt"
io = open(filename, "w")
@printf(io, "#   frame \t i1 \t i2 \t j1 \t j2 \t Max(Gc)\n")
# End of output preparation

println("frame \t i1 \t i2 \t j1 \t j2 \t Max(Gc)")

start_time = time_ns()
for frame = begin_frame:increment_num_frames:end_frame
    # single frame
    coor = reshape(t["atomname CA"].xyz[frame, :], (3, n_atoms))'
    """
        @. indicates operator here is not working on vector.
        slicing array make a copy of sub-array => use @view macro will reduce
            memory allocation and speedup computational time
    """
    Rcoor = @. 0.5 * (@view(coor[1:n_atoms-1, :]) + @view(coor[2:n_atoms, :]))
    dRcoor = @. @view(coor[2:n_atoms, :]) - @view(coor[1:n_atoms-1, :])

    # len_coor = size(Rcoor)[1]
    R_diff = reshape(
        [
            @. @view(Rcoor[i, :]) - @view(Rcoor[j, :]) for j = 1:len_coor,
            i = 1:len_coor
        ],
        (len_coor, len_coor),
    )
    dR_cross = reshape(
        [
            cross(@view(dRcoor[i, :]), @view(dRcoor[j, :])) for
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
    contact_list = []
    pair_dis = pairwise(euclidean, coor, dims = 1)
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
    results = zeros(Float64, (nthreads() * space, 5))
    @threads for cl in contact_list
        i1, i2 = cl
        for j1 = 1:i1-10, j2 = j1+10:i1-1
            @fastmath @inbounds res =
                abs(sum(@view(dot_cross_matrix[i1:i2-1, j1:j2-1])) / (4 * pi))
            if res >= results[threadid()*space, 1]
                results[threadid()*space, :] = [res, i1, i2, j1, j2]
            end
        end

        for j1 = i2+1:len_coor-10, j2 = j1+10:len_coor
            @fastmath @inbounds res =
                abs(sum(@view(dot_cross_matrix[i1:i2-1, j1:j2-1])) / (4 * pi))
            if res >= results[threadid()*space, 1]
                results[threadid()*space, :] = [res, i1, i2, j1, j2]
            end
        end
    end
    idx_max_gc = argmax(results[:, 1])
    max_gc, idx_i1, idx_i2, idx_j1, idx_j2 = results[idx_max_gc, :]
    if Int64(idx_i1) == 0
        # no loop in this case, then all variables here are 0. just compare idx_i1 is enough
        @printf(
            "%d \t %d \t %d \t %d \t %d \t %.3f\n",
            frame,
            Int64(idx_i1),
            Int64(idx_i2),
            Int64(idx_j1),
            Int64(idx_j2),
            max_gc
        )
        @printf(
            io,
            "%8d \t %3d \t %3d \t %3d \t %3d \t %.3f\n",
            frame,
            Int64(idx_i1),
            Int64(idx_i2),
            Int64(idx_j1),
            Int64(idx_j2),
            max_gc
        )
    else

        @printf(
            "%d \t %d \t %d \t %d \t %d \t %.3f\n",
            frame,
            resids[Int64(idx_i1)],
            resids[Int64(idx_i2)],
            resids[Int64(idx_j1)],
            resids[Int64(idx_j2)],
            max_gc
        )
        @printf(
            io,
            "%8d \t %3d \t %3d \t %3d \t %3d \t %.3f\n",
            frame,
            resids[Int64(idx_i1)],
            resids[Int64(idx_i2)],
            resids[Int64(idx_j1)],
            resids[Int64(idx_j2)],
            max_gc
        )
    end
end
close(io)
end_time = time_ns()
dt = (end_time - start_time) / 10^9
println("Execution time: ", dt, "(s)")
