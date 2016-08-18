#!/usr/bin/julia --color=yes

using ArgParse

using JLD
using Multilane
using POMDPs
using MCTS
using DataFrames
using DataFramesMeta
using Plots
using StatPlots

exception = Nullable{Any}()

s = ArgParseSettings()

@add_arg_table s begin
    "--show", "-s"
        help = "show results; print stats and solvers"
        action = :store_true
    "--unicode", "-u"
        help = "show unicode plot"
        action = :store_true
    "--plot", "-p"
        help = "plot paretto curves"
        action = :store_true
    "--spreadsheet"
        help = "[NOT IMPLEMENTED] save stats as a csv in /tmp/ and open with xdg-open"
        action = :store_true
    "filename"
        help = "file name"
        nargs = '+' # will be + at some point
    "--solvers"
        help = "solvers to be included"
        nargs = '+'
    "--save-combined"
        help = "save results to a new file"
        action = :store_true
    "--check-crashes"
        help = "print out simulations that have crashes"
        action = :store_true
end

args = parse_args(ARGS, s)

results = load(args["filename"][1])

for f in args["filename"][2:end]
    new_results = load(f)
    results = merge_results!(results, new_results)
end

stats = results["stats"]

mean_performance = by(stats, [:solver_key, :lambda, :problem_key, :solver_problem_key]) do df
    DataFrame(steps_in_lane=mean(df[:steps_in_lane]),
              nb_brakes=mean(df[:nb_brakes]),
              time_to_lane=mean(df[:time_to_lane]),
              steps=mean(df[:steps]),
              test_key=first(df[:test_key])
              )
end

solvers = args["solvers"]
if isempty(solvers)
    solvers = unique(mean_performance[:solver_key])
end

mean_performance = @where(mean_performance,
                          collect(Bool[s in solvers for s in :solver_key]))

mean_performance[:brakes_per_sec] = mean_performance[:nb_brakes]./mean_performance[:time_to_lane]

if args["show"]
    println(mean_performance)
    println()
    println(results["param_table"])
end

if args["unicode"] || args["plot"]
    try
        if args["unicode"]
            unicodeplots()
        else
            pyplot()
        end
        if haskey(results, "tests")
            for g in groupby(mean_performance, :test_key)
                if size(g,1) > 1
                    plot!(g, :time_to_lane, :brakes_per_sec, group=:test_key)
                else
                    scatter!(g, :time_to_lane, :brakes_per_sec, group=:test_key)
                end
            end
            gui()
        else # old way, before test sets
            for g in groupby(mean_performance, :solver_key)
                if size(g,1) > 1
                    plot!(g, :time_to_lane, :brakes_per_sec, group=:solver_key)
                else
                    scatter!(g, :time_to_lane, :brakes_per_sec, group=:solver_key)
                end
            end
            gui()
        end
    catch ex
        warn("Plot could not be displayed:")
        println(ex)
        exception = Nullable{Any}(ex)
        # rethrow(ex)
    end
end

if args["save-combined"]
    filename = string("combined_", Dates.format(Dates.now(),"u_d_HH_MM"), ".jld")
    save(filename, results)
    println("combined results saved to $filename")
end

if !isnull(exception)
    rethrow(get(exception))
end
