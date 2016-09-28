using Multilane
using MCTS
using RobustMCTS
using GenerativeModels
using POMDPToolbox
using JLD
using POMDPs
using DataFrames
using Plots

N=500

filename = "initials_Jul_18_10_19.jld"
# filename = "initials_Jul_4_17_11.jld"
# filename = "initials_Jul_7_13_02.jld"
initials = load(filename)
problems = initials["problems"]
initial_states = initials["initial_states"]

# Calculate single points
point_solvers = Dict{String, Solver}(
    "random" => RandomSolver(),
    "heuristic" => SimpleSolver()
)

point_results = evaluate(Dict(take(problems,1)), initial_states, point_solvers, parallel=true, N=N)

dpws = DPWSolver(depth=20,
                 n_iterations=2000,
                 exploration_constant=50.0,
                 k_state=4.0,
                 alpha_state=1/8,
                 rollout_solver=SimpleSolver()) 

rsolver = RobustMCTSSolver(
    depth=20,
    c = 50.0,
    n_iterations=10000,
    k_state=4.0,
    alpha_state=1/8,
    k_nature=4.0,
    alpha_nature=1/8,
    rollout_solver=SimpleSolver(),
    rollout_nature=StochasticBehaviorNoCrashMDP(collect(values(problems))[1]))

curve_solvers = Dict{String, Solver}(
    "dpw_2k" => dpws,
    "robust_10k" => RobustMLSolver(rsolver),
    "single_behavior_2k" => 
        SingleBehaviorSolver(dpws,
                 IDMMOBILBehavior("normal", 30.0, 10.0, 1))
)

curve_results = evaluate(problems, initial_states, curve_solvers, parallel=true, N=N)

results = merge_results!(curve_results, point_results)

println(results["stats"])

filename = string("results_", Dates.format(Dates.now(),"u_d_HH_MM"), ".jld")
results["histories"] = nothing
save(filename, results)
println("results saved to $filename")

stats = results["stats"]
mean_performance = by(stats, :solver_key) do df
    by(df, :lambda) do df
        DataFrame(steps_in_lane=mean(df[:steps_in_lane]),
                  nb_brakes=mean(df[:nb_brakes]),
                  time_to_lane=mean(df[:time_to_lane]),
                  steps=mean(df[:steps]),
                  )
    end
end

unicodeplots()
for g in groupby(mean_performance, :solver_key)
    if size(g,1) > 1
        # plot!(g, :steps_in_lane, :nb_brakes, group=:solver_key)
        plot!(g, :time_to_lane, :nb_brakes, group=:solver_key)
    else
        # scatter!(g, :steps_in_lane, :nb_brakes, group=:solver_key)
        scatter!(g, :time_to_lane, :nb_brakes, group=:solver_key)
    end
end
gui()

