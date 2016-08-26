using Multilane
using MCTS
using POMCP

using ArgParse

s = ArgParseSettings()

@add_arg_table s begin
    "dir"
        help = "data directory - will look here for results_list.txt"
        nargs = 1
end

args = parse_args(ARGS, s)

files = []
open(joinpath(first(args["dir"]), "results_list.txt")) do list
    for f in eachline(list)
        push!(files, chomp(f))
    end
end

tic()
results = gather_results(files, save_file=Nullable(string("combined_results_", Dates.format(Dates.now(),"u_d_HH_MM"), ".jld")))
toc()
