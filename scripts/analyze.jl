#!/usr/bin/julia

using ArgParse
using JLD
using Multilane
using POMDPs
using MCTS
using DataFrames

s = ArgParseSettings()

@add_arg_table s begin
    "--show", "-s"
        help = "show results; print stats and solvers"
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





if args["show"]
    println("""

        =============
        ## Solvers ##
        =============

        """)
    for (k,v) in results["solvers"]
        println("$k:")
        println(v)
        println()
    end

    println("""

        ================
        ## Statistics ##
        ================

        """)
    showall(results["stats"])
    println()
end
