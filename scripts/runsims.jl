#!/usr/bin/julia

tic()

using ArgParse

using JLD
using Multilane
using POMDPs
using MCTS
using DataFrames
using DataFramesMeta

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
if any(isa(r, RemoteException) || !isnull(r.exception) for r in sims)
    print_with_color(:red, """
        Error in simulations. Run the following to debug:

        ARGS = $ARGS
        include("$(@__FILE__)")
        """)
        # julia -i $(@__FILE__) $(first(args["objectfile"])) $(first(args["listfile"]))
    println()
end
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
objects["histories"]=nothing

save(resultsfile, objects)
println("Done running $(nrow(stats)) tests.")
toc()
