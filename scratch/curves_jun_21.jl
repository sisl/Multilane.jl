using Multilane
using MCTS
using GenerativeModels
using POMDPToolbox
using JLD
using POMDPs

# NOTE: assumes 4 lanes, desired lane is lane 4
desired_lane_reward = 10.

lambdas = Float64[0.1, 1., 10., 100., 1000.]

nb_lanes = 4 # XXX assumption

rmodels = Multilane.NoCrashRewardModel[
                Multilane.NoCrashRewardModel(desired_lane_reward*lambda,
                                             desired_lane_reward,2.0,nb_lanes)
                for lambda in lambdas]


nb_lanes = 4
pp = PhysicalParam(nb_lanes,lane_length=100.) #2.=>col_length=8\n",
_discount = 1.
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

dpws = DPWSolver(depth=10,
                 n_iterations=100,
                 exploration_constant=100.0,
                 rollout_solver=SimpleSolver()) 

curve_solvers = Dict{UTF8String, Solver}(
    "dpw" => dpws,
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

results = merge!(curve_results, point_results)

println(results["stats"])

filename = string("results_", Dates.format(Dates.now(),"u_d_HH_MM"), ".jld")
save(filename, results)
println("results saved to $filename")
