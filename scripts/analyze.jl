#!/usr/bin/env julia --color=yes

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
        nargs = '*'
    "--solvers"
        help = "solvers to be included"
        nargs = '+'
    "--tests"
        help = "tests to be included"
        nargs = '+'
    "--save-combined"
        help = "save results to a new file"
        action = :store_true
    "--check-crashes"
        help = "print out simulations that have crashes"
        action = :store_true
    "--list", "-l"
        help = "list note and test keys"
        action = :store_true
end

args = parse_args(ARGS, s)

if length(args["filename"]) > 0
    filename = first(args["filename"])
else
    files = readdir(".")
    latest = 0.0
    latest_file = ""
    for f in files
        if ctime(f) > latest
            latest = ctime(f)
            latest_file = f
        end
    end
    println("loading from most recent file: $latest_file")
    filename = latest_file
end

results = load(filename)

if args["list"]
    println("$filename note:")
    println(get(results, "note", "<none>"))
end

for f in args["filename"][2:end]
    new_results = load(f)
    try
        results = merge_results!(results, new_results)
    catch ex
        warn("Careful merge of data failed, there may be inconsistent data")
        results = merge_results!(results, new_results, careful=false)
    end
    if args["list"]
        println("$filename note:")
        println(get(results, "note", "<none>"))
    end
end

stats = results["stats"]

mean_performance = by(stats, [:solver_key, :lambda, :problem_key, :solver_problem_key, :test_key]) do df
    DataFrame(steps_in_lane=mean(df[:steps_in_lane]),
              nb_brakes=mean(df[:nb_brakes]),
              time_to_lane=mean(df[:time_to_lane]),
              steps=mean(df[:steps]),
              )
end

solvers = args["solvers"]
if isempty(solvers)
    solvers = unique(mean_performance[:solver_key])
end
tests = args["tests"]
if isempty(tests)
    tests = unique(mean_performance[:test_key])
end

if args["list"]
    println()
    println("Avaialable Tests:")
    for t in unique(mean_performance[:test_key])
        println(t)
    end
    println()
end

mean_performance = @where(mean_performance,
                          collect(Bool[s in solvers for s in :solver_key]))
mean_performance = @where(mean_performance,
                          collect(Bool[t in tests for t in :test_key]))

# @show mean_performance

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
            # plotlyjs()
            # pgfplots()
        end
        if haskey(results, "tests")
            for g in groupby(mean_performance, :test_key)
                if size(g,1) > 1
                    plot!(g, :time_to_lane, :nb_brakes, group=:test_key)
                else
                    scatter!(g, :time_to_lane, :nb_brakes, group=:test_key)
                end
            end
            gui()
        else # old way, before test sets
            for g in groupby(mean_performance, :solver_key)
                if size(g,1) > 1
                    plot!(g, :time_to_lane, :nb_brakes, group=:solver_key)
                else
                    scatter!(g, :time_to_lane, :nb_brakes, group=:solver_key)
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
