using Multilane
using MCTS
using POMCP

using ArgParse

s = ArgParseSettings()

@add_arg_table s begin
    "dir"
        help = "data directory - will look here for results_list.txt"
        nargs = 1
        required = false
end

args = parse_args(ARGS, s)

if length(args["dir"]) > 0
    dir = first(args["dir"])
else
    parent = get(ENV, "SCRATCH", tempdir())
    dirs = readdir(parent)
    latest = 0.0
    latest_dir = ""
    for d in dirs
        abspath = joinpath(parent, d)
        if ctime(abspath) > latest
            latest = ctime(abspath)
            latest_dir = abspath
        end
    end
    dir = latest_dir
end

println("Looking at $(joinpath(dir, "results_list.txt"))")
files = []
open(joinpath(dir, "results_list.txt")) do list
    for f in eachline(list)
        push!(files, chomp(f))
    end
end

filename = string("combined_results_", Dates.format(Dates.now(),"u_d_HH_MM"), ".jld")
results = gather_results(files, save_file=Nullable(filename))
println("saved to $filename")
