#!/usr/bin/julia --color=yes

# THIS IS OLD - USE COMBINED INSTEAD

using ArgParse

using JLD
using Multilane
using POMDPs
using MCTS
using DataFrames
using DataFramesMeta
using PGFPlots

function print_tex(p, name)
    print_tex(plot(p), name)
end

function print_tex(p::TikzPictures.TikzPicture, name)
    println("""

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % $name
        """)
    println(p.data)
    println("""
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        """)
end

function show_svg(p)
    f = string(tempname(), ".svg")
    PGFPlots.save(f, plot(p))
    @spawn run(`display -alpha off $f`)
end

bg = standard_uniform(1.0, correlation=0.75)

N = 1000

rng = MersenneTwister(123)

bs = Array(BehaviorModel, N)
for i in 1:N
    bs[i] = rand(rng, bg)
end

v0s_75 = Float64[bs[i].p_idm.v0 for i in 1:N]
Ts_75 = Float64[bs[i].p_idm.T for i in 1:N]


bg = standard_uniform(1.0, correlation=true)

rng = MersenneTwister(123)

bs = Array(BehaviorModel, N)
for i in 1:N
    bs[i] = rand(rng, bg)
end

v0s_1 = Float64[bs[i].p_idm.v0 for i in 1:N]
Ts_1 = Float64[bs[i].p_idm.T for i in 1:N]


bg = standard_uniform(1.0, correlation=false)

rng = MersenneTwister(123)

bs = Array(BehaviorModel, N)
for i in 1:N
    bs[i] = rand(rng, bg)
end

v0s_0 = Float64[bs[i].p_idm.v0 for i in 1:N]
Ts_0 = Float64[bs[i].p_idm.T for i in 1:N]

axis_args = Dict{Symbol, Any}(
    # :width=>"0.5\\columnwidth"
    :width=>"5cm"
)

scatter_args = Dict{Symbol, Any}(
    :markSize=>1
)

pushPGFPlotsPreamble("\\usepackage{siunitx}")

g = GroupPlot(2, 2, groupStyle = "horizontal sep = 0.075\\columnwidth, vertical sep = 0.15\\columnwidth")
push!(g, Axis(Plots.Scatter(v0s_0, Ts_0; scatter_args...); title="Scenario 1, \$\\rho=0\$", ylabel="\$T\$ (\\si{s})", axis_args...))
push!(g, Axis(Plots.Scatter(v0s_1, Ts_1; scatter_args...); title="Scenario 2, \$\\rho=1\$", xlabel="\$v_0\$ (\\si{m/s})", axis_args...))
push!(g, Axis(Plots.Scatter(v0s_75, Ts_75; scatter_args...); title="Scenario 3, \$\\rho=0.75\$", xlabel="\$v_0\$ (\\si{m/s})", ylabel="\$T\$ (\\si{s})", axis_args...))
    

# print_tex(g, "SCATTER")

PGFPlots.save("/home/zach/Devel/Behavior-aware_lane_changing/acc_17_paper/fig/scatter.tex", g)

# show_svg(g)
