#!/usr/bin/julia

using ArgParse

using JLD
using Multilane
using POMDPs
using MCTS
using DataFrames
using DataFramesMeta
using Plots
using StatPlots

s = ArgParseSettings()

@add_arg_table s begin
    "objectfile"
        help = "jld file containing all of the objects"
        nargs = 1
    "listfile"
        help = "jld file containing the stats table of the sims to be run"
        nargs = 1
    "--progress"
        help = "print progress information"
        action = :store_true
end

args = parse_args(ARGS, s)

objects = load(first(args["objectfile"]))
listfile = first(args["listfile"])
list = load(listfile)
stats = list["stats"]
metrics = get(list, "metrics", [])

sims = run_simulations(stats, objects, progress=args["progress"])
fill_stats!(stats, objects, sims)

objects["stats"] = stats

dir = dirname(listfile)
b = basename(listfile)
if contains(b, "list")
    b = replace(b, "list", "results")
else
    b = string("results_", b)
end
resultsfile = joinpath(dir, b)
save(resultsfile, objects)
