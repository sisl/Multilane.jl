#!/usr/bin/julia --color=yes

using ArgParse

using JLD
using Multilane
using POMDPs
using MCTS
using DataFrames
using Plots

s = ArgParseSettings()

@add_arg_table s begin
    "--show", "-s"
        help = "show results; print stats and solvers"
        action = :store_true
    "--unicode", "-u"
        help = "show unicode plot"
        action = :store_true
    "--performance"
        help = "[NOT IMPLEMENTED] show average performance"
        action = :store_true
    "--plot", "-p"
        help = "[NOT IMPLEMENTED] plot paretto curves"
        action = :store_true
    "--spreadsheet"
        help = "[NOT IMPLEMENTED] save stats as a csv in /tmp/ and open with xdg-open"
        action = :store_true
    "filename"
        help = "file name"
        nargs = 1 # will be + at some point
end

args = parse_args(ARGS, s)

results = load(args["filename"][1])

stats = results["stats"]

mean_performance = by(stats, :solver_key) do df
    by(df, :lambda) do df
        DataFrame(steps_in_lane=mean(df[:steps_in_lane]),
                  nb_brakes=mean(df[:nb_brakes])
                  )
    end
end

# if args["plot"]
#     unicodeplots()
#     plot(
# end

if args["show"]
#     println("""
# 
#         =============
#         ## Solvers ##
#         =============
# 
#         """)
#     for (k,v) in results["solvers"]
#         println("$k:")
#         println(v)
#         println()
#     end
# 
#     println("""
# 
#         ================
#         ## Statistics ##
#         ================
# 
#         """)
    println(mean_performance)
    println()
end

if args["unicode"]
    unicodeplots()
    for g in groupby(mean_performance, :solver_key)
        if size(g,1) > 1
            plot!(g, :steps_in_lane, :nb_brakes, group=:solver_key)
        else
            scatter!(g, :steps_in_lane, :nb_brakes, group=:solver_key)
        end
    end
    gui()
end
