#!/usr/bin/julia --color=yes

# THIS IS OLD USE COMBINED INSTEAD

using ArgParse

using JLD
using Multilane
using POMDPs
using MCTS
using DataFrames
using DataFramesMeta
using PGFPlots

function curve_plot(mp::AbstractDataFrame, test_key::AbstractString; label=test_key, style="")
    data = sort!(@where(mp, :test_key.==test_key), cols=[:lambda])
    Plots.Linear(data[:time_to_lane], data[:nb_brakes], legendentry=label, style=style)
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

function print_tex(p::Axis, name)
    print_tex(plot(p), name)
end

function show_svg(p::Axis)
    f = string(tempname(), ".svg")
    PGFPlots.save(f, plot(p))
    @spawn run(`display -alpha off $f`)
end

function ttl_at_brake(mp, test_key, nb_brakes)
    data = sort!(@where(mp, :test_key.==test_key), cols=[:lambda])
    i = findfirst(data[:nb_brakes] .<= nb_brakes)
    ttl = data[:time_to_lane]
    nb = data[:nb_brakes]
    return ttl[i-1] + (ttl[i]-ttl[i-1])*(nb_brakes-nb[i-1])/(nb[i]-nb[i-1])
end

function brake_at_ttl(mp, test_key, t)
    data = sort!(@where(mp, :test_key.==test_key), cols=[:lambda])
    ttl = data[:time_to_lane]
    i = findfirst(ttl .>= t)
    nb = data[:nb_brakes]
    return nb[i-1] + (nb[i]-nb[i-1])*(t-ttl[i-1])/(ttl[i]-ttl[i-1])
end


filenames = ["combined_results_Sep_2_11_31.jld", "combined_results_Sep_2_11_52.jld", "combined_results_Sep_8_21_57.jld"]
filenames = [joinpath(Pkg.dir("Multilane"), "data", f) for f in filenames]

if !isdefined(:results)
    results = load(first(filenames))

    for f in filenames[2:end]
        new_results = load(f)
        try
            results = merge_results!(results, new_results)
        catch ex
            warn("Careful merge of data failed, there may be inconsistent data")
            results = merge_results!(results, new_results, careful=false)
        end
    end
end

stats = results["stats"]

mp = by(stats, [:solver_key, :lambda, :problem_key, :solver_problem_key, :test_key]) do df
    DataFrame(steps_in_lane=mean(df[:steps_in_lane]),
              nb_brakes=mean(df[:nb_brakes]),
              time_to_lane=mean(df[:time_to_lane]),
              steps=mean(df[:steps]),
              )
end

@show unique(mp[:test_key])

# first plot = uniform
# a = Axis([
#     curve_plot(mp, "upper_bound_unif", label="Omniscience bound"),
#     curve_plot(mp, "assume_normal_unif", label="Assume all normal"),
# ], xmin=0,
# xmax=30,
# ymin=0,
# ymax=1.5,
# xlabel="Time to change lanes",
# ylabel="Braking actions per episode",
# style="grid=both", title="Behaviors Uniformly Distributed")
# 
# # show_svg(a)
# 
# print_tex(a, "UNIFORM")


# uniform with pomdp solutions
a = Axis([
    curve_plot(mp, "upper_bound_unif", label="Omniscient", style="dashed"),
    # Plots.Command("\\only<2->{"),
    curve_plot(mp, "assume_normal_unif", label="Single assumed behavior"),
    # Plots.Command("}"),
    curve_plot(mp, "mlmpc_unif", label="Most likely behavior"),
    # Plots.Command("\\only<3->{"),
    curve_plot(mp, "pomcp_unif", label="POMCP"),
    # Plots.Command("}")
], xmin=0,
xmax=30,
ymin=0,
ymax=1.5,
xlabel="Time to target lane",
ylabel="Braking actions per episode",
style="grid=both, legend cell align=left",
# title="Scenario 1: \$\\rho = 0\$"
)

@show ub = ttl_at_brake(mp, "upper_bound_unif", 0.5)
@show an = ttl_at_brake(mp, "assume_normal_unif", 0.5)
@show an - ub
@show (an - ub)/ub

@show bub = brake_at_ttl(mp, "upper_bound_unif", 10.0)
@show ban = brake_at_ttl(mp, "assume_normal_unif", 10.0)
@show ban - bub
@show (ban - bub)/bub

# print_tex(a, "UNIFORM SOLVERS")

PGFPlots.save("/home/zach/Devel/Behavior-aware_lane_changing/acc_17_paper/fig/uniform.tex", a, include_preamble=false)
# show_svg(a)

# correlate with pomdp solutions
a = Axis([
    curve_plot(mp, "upper_bound", label="Omniscient", style="dashed"),
    # Plots.Command("\\only<2->{"),
    curve_plot(mp, "assume_normal", label="Single assumed behavior"),
    # Plots.Command("}"),
    curve_plot(mp, "mlmpc", label="Most likely behavior"),
    # Plots.Command("\\only<3->{"),
    curve_plot(mp, "pomcp", label="POMCP"),
    # Plots.Command("}")
], xmin=0,
xmax=30,
ymin=0,
ymax=1.5,
xlabel="Time to target lane",
ylabel="Braking actions per episode",
style="grid=both, legend cell align=left", 
# title="Scenario 2: \$\\rho = 1\$"
)

# calculate the ttl for 0.5 brakes
# between 4 and 5


# print_tex(a, "CORRELATED SOLVERS")

PGFPlots.save("/home/zach/Devel/Behavior-aware_lane_changing/acc_17_paper/fig/correlated.tex", a, include_preamble=false)
# show_svg(a)


# cors = [0.5, 0.75, 0.9]
cors = [0.75]
for cor in cors
    a = Axis([
        curve_plot(mp, @sprintf("upper_bound_%03d", 100*cor), label="Omniscient", style="dashed"),
        # Plots.Command("\\only<2->{"),
        curve_plot(mp, @sprintf("assume_normal_%03d", 100*cor), label="Single assumed behavior"),
        # Plots.Command("}"),
        curve_plot(mp, @sprintf("mlmpc_%03d", 100*cor), label="Most likely behavior"),
        # Plots.Command("\\only<3->{"),
        curve_plot(mp, @sprintf("pomcp_%03d", 100*cor), label="POMCP"),
        # Plots.Command("}")
    ], xmin=0,
    xmax=30,
    ymin=0,
    ymax=1.5,
    xlabel="Time to target lane",
    ylabel="Braking actions per episode",
    style="grid=both, legend cell align=left",
    # title="Scenario 3: \$\\rho = $cor\$"
    )

    # print_tex(a, "CORRELATED $cor")

    PGFPlots.save(@sprintf("/home/zach/Devel/Behavior-aware_lane_changing/acc_17_paper/fig/correlated_%03d.tex", 100*cor), a, include_preamble=false)

    # show_svg(a)
end
