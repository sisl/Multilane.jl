#!/usr/bin/julia

using ArgParse

using JLD
using Multilane
using POMDPs
using MCTS
using DataFrames
using DataFramesMeta
using PGFPlots
pushPGFPlotsPreamble("\\usepackage{siunitx}")

function curve_plot(mp::AbstractDataFrame,
                   test_key::AbstractString;
                   label=replace(test_key,"_","\\_"),
                   style=nothing,
                   )
    data = sort!(@where(mp, :test_key.==test_key), cols=[:lambda])
    # Plots.Linear(data[:time_to_lane], data[:nb_brakes], legendentry=label, style=style)
    ttl = data[:time_to_lane]
    nb = data[:nb_brakes]
    ttl_sem = data[:ttl_sem]
    nb_sem = data[:nb_sem]
    mat = cat(2, ttl, nb, ttl_sem, nb_sem, ttl_sem, nb_sem)'
    Plots.ErrorBars(mat, legendentry=label, style=style)
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

function show_svg(p)
    f = tempname()
    @show f
    PGFPlots.save(string(f, ".svg"), plot(p))
    # compile = `pdflatex -batch $f.tex`
    # println("running $compile")
    # run(compile)
    @spawn run(`display -alpha off $f.svg`)
    # PGFPlots.save(string(f, ".tex"), plot(p))
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


# filenames = ["combined_results_Sep_2_11_31.jld", "combined_results_Sep_2_11_52.jld", "combined_results_Sep_8_21_57.jld"]
filenames = ["combined_results_Sep_12_11_51.jld", "combined_results_Sep_16_16_46.jld", "combined_results_Sep_16_17_51.jld"]
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
    DataFrame(
                  steps_in_lane=mean(df[:steps_in_lane]),
                  nb_brakes=mean(df[:nb_brakes]),
                  time_to_lane=mean(df[:time_to_lane]),
                  steps=mean(df[:steps]),
                  nb_sem=std(df[:nb_brakes])/sqrt(length(df[:nb_brakes])),
                  ttl_sem=std(df[:time_to_lane])/sqrt(length(df[:time_to_lane]))
              )
end

@show unique(mp[:test_key])

N = 1000
rng = MersenneTwister(123)
axis_args = Dict{Symbol, Any}(
    # :width=>"0.5\\columnwidth"
    :width=>"4cm",
    :style=>"xshift=4cm, yshift=3cm, axis background/.style={fill=white}",
    # :ylabel=>"\$T\$ (\\si{s})",
    # :xlabel=>"\$v_0\$ (\\si{m/s})"
)
scatter_args = Dict{Symbol, Any}(
    :markSize=>1,
    # :style=>"fill opacity=1"
)

g = GroupPlot(1,3)

bg = standard_uniform(1.0, correlation=false)
bs = Array(BehaviorModel, N)
for i in 1:N
    bs[i] = rand(rng, bg)
end
v0s_0 = Float64[bs[i].p_idm.v0 for i in 1:N]
Ts_0 = Float64[bs[i].p_idm.T for i in 1:N]

a = Axis([
    curve_plot(mp, "upper_bound_unif", label=nothing, style="dashed"),
    curve_plot(mp, "assume_normal_unif", label=nothing),
    curve_plot(mp, "mlmpc_unif", label=nothing),
    curve_plot(mp, "pomcp_unif", label=nothing)
], xmin=0,
xmax=30,
ymin=0,
ymax=1.5,
# xlabel="Time to target lane",
ylabel="Braking actions per episode",
style="grid=both",
# title = "Scenario 1: \$\\rho = 0\$"
)

s = Axis(Plots.Scatter(v0s_0, Ts_0; scatter_args...); title="Scenario 1: \$\\rho=0\$", axis_args...)

push!(g,a)
# show_svg([a,s])
# print_tex(a, "UNIFORM SOLVERS")
PGFPlots.save("/home/zach/Devel/Behavior-aware_lane_changing/acc_17_paper/fig/uniform.tex", [a,s], include_preamble=false)


@show ub = ttl_at_brake(mp, "upper_bound_unif", 0.5)
@show an = ttl_at_brake(mp, "assume_normal_unif", 0.5)
@show an - ub
@show (an - ub)/an

@show bub = brake_at_ttl(mp, "upper_bound_unif", 10.0)
@show ban = brake_at_ttl(mp, "assume_normal_unif", 10.0)
@show ban - bub
@show (ban - bub)/ban

bg = standard_uniform(1.0, correlation=true)
bs = Array(BehaviorModel, N)
for i in 1:N
    bs[i] = rand(rng, bg)
end
v0s_1 = Float64[bs[i].p_idm.v0 for i in 1:N]
Ts_1 = Float64[bs[i].p_idm.T for i in 1:N]

a = Axis([
    curve_plot(mp, "upper_bound", label=nothing, style="dashed"),
    curve_plot(mp, "assume_normal", label=nothing),
    curve_plot(mp, "mlmpc", label=nothing),
    curve_plot(mp, "pomcp", label=nothing)
], xmin=0,
xmax=30,
ymin=0,
ymax=1.5,
# xlabel="Time to target lane",
ylabel="Braking actions per episode",
style="grid=both",
# title="Scenario 2: \$\\rho = 1\$"
)

s = Axis(Plots.Scatter(v0s_1, Ts_1; scatter_args...); title="Scenario 2: \$\\rho=1\$", axis_args...)

push!(g,a)
# show_svg([a,s])
# print_tex(a, "UNIFORM SOLVERS")
PGFPlots.save("/home/zach/Devel/Behavior-aware_lane_changing/acc_17_paper/fig/correlated.tex", [a,s], include_preamble=false)

bg = standard_uniform(1.0, correlation=0.75)
bs = Array(BehaviorModel, N)
for i in 1:N
    bs[i] = rand(rng, bg)
end
v0s_75 = Float64[bs[i].p_idm.v0 for i in 1:N]
Ts_75 = Float64[bs[i].p_idm.T for i in 1:N]

a = Axis([
    curve_plot(mp, "upper_bound_075", label="Omniscient", style="dashed"),
    curve_plot(mp, "assume_normal_075", label="SAB"),
    curve_plot(mp, "mlmpc_075", label="MLMPC"),
    curve_plot(mp, "pomcp_075", label="POMCP")
], xmin=0,
xmax=30,
ymin=0,
ymax=1.5,
xlabel="Time to target lane",
ylabel="Braking actions per episode",
style="grid=both, legend columns=2, legend style={at={(0.5, -0.2)},anchor=north}",
# title="Scenario 3: \$\\rho = 0.75\$"
)

s = Axis(Plots.Scatter(v0s_75, Ts_75; scatter_args...); title="Scenario 3: \$\\rho=0.75\$", axis_args...)

push!(g,a)
# show_svg([a,s])
# print_tex(a, "UNIFORM SOLVERS")
PGFPlots.save("/home/zach/Devel/Behavior-aware_lane_changing/acc_17_paper/fig/correlated_075.tex", [a,s], include_preamble=false)

# show_svg(g)
