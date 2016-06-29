using Multilane
using MCTS
using GenerativeModels
using POMDPToolbox
using JLD
using POMDPs
using DataFrames
using Plots

# NOTE: assumes 4 lanes, desired lane is lane 4
desired_lane_reward = 10.

lambdas = Float64[0.01, 0.1, 1., 10., 100.]

nb_lanes = 4 # XXX assumption

rmodels = Multilane.NoCrashRewardModel[
                Multilane.NoCrashRewardModel(desired_lane_reward*lambda,
                                             desired_lane_reward,1.5,nb_lanes)
                for lambda in lambdas]


nb_lanes = 4
pp = PhysicalParam(nb_lanes,lane_length=100.) #2.=>col_length=8\n",
_discount = 0.9
nb_cars=10
dmodel = NoCrashIDMMOBILModel(nb_cars, pp)

N = 100

mdp = NoCrashMDP(dmodel, rmodels[1], _discount);
isrng = MersenneTwister(123)
initial_states = MLState[initial_state(mdp, isrng) for i in 1:N]

# Calculate single points
point_solvers = Dict{UTF8String, Solver}(
    "random" => RandomSolver(),
    "heuristic" => SimpleSolver()
)

point_results = evaluate([mdp], initial_states, point_solvers, parallel=true)

dpws = DPWSolver(depth=20,
                 n_iterations=100,
                 exploration_constant=100.0,
                 k_state=4.0,
                 alpha_state=1/8,
                 rollout_solver=SimpleSolver()) 

curve_solvers = Dict{UTF8String, Solver}(
    "dpw" => dpws,
    # "robust" => 
    #     Robust
    "single_behavior" => 
        SingleBehaviorSolver(dpws,
                 IDMMOBILBehavior("normal", 30.0, 10.0, 1))
)

curve_problems = []
for i in 1:length(lambdas)
    rmodel = rmodels[i]
    mdp = NoCrashMDP(dmodel, rmodel, _discount);
    push!(curve_problems, mdp)
end

curve_results = evaluate(curve_problems, initial_states, curve_solvers, parallel=true)

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

