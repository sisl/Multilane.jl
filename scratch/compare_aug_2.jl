using Multilane
using MCTS
using RobustMCTS
using POMCP
using GenerativeModels
using POMDPToolbox
using JLD
using POMDPs
using DataFrames
using Plots

N=500

problems = Dict{UTF8String, Any}()
key_rng = MersenneTwister(123)

isrng = MersenneTwister(123)
init_phys_states = [MLPhysicalState(initial_state(mdp, isrng)) for i in 1:N]

params = Dict{UTF8String, Any}()
params["lambda"] = Float64[0.1, 1., 10., 17.78, 31.62, 56.23, 100., 1000.]
params["p_normal"] = Float64[0.2, 1.0]

initial_relevant = ["p_normal"]

params_to_problem = Dict{Dict{UTF8String,Any},UTF8String}()
problem_to_initials = Dict{UTF8String,AbstractVector}()

nb_lanes = 4
desired_lane_reward = 10.
rmodel = NoCrashRewardModel(desired_lane_reward*10., desired_lane_reward,2.5,nb_lanes)

pp = PhysicalParam(nb_lanes,lane_length=100.)

normal_behavior = IDMMOBILBehavior("normal", pp.v_med, pp.l_car, 1)
behaviors=IDMMOBILBehavior[
                    normal_behavior,
                    IDMMOBILBehavior("cautious", pp.v_slow+0.5, pp.l_car, 2),
                    IDMMOBILBehavior("aggressive", pp.v_fast, pp.l_car, 3)
                    ]

_discount = 1.0
nb_cars = 10
dmodel = NoCrashIDMMOBILModel(nb_cars, pp,
                              behaviors=behaviors,
                              behavior_probabilities=WeightVec([0.2, 0.4, 0.4]),
                              lane_terminate=false)

base_problem = NoCrashMDP(dmodel, rmodel, _discount)

#=
param_keys = collect(keys(params))
param_values = collect(values(params)) # assume that these are ordered the same
for vals in Iterators.product(param_values...)
    these_params = Dict{UTF8String, Any}([(param_keys[i], vals[i]) for i in length(param_keys)])
    
    problem = deepcopy(base_problem)

    for (param_key,val) in these_params 
        if param_key == "lambda"
            #XXX adjust problem
        else if param_key == "p_normal"
            #XXX adjust problem
        else
            warn("Ignored key $key")
        end
    end
    problem_key = randstring(key_rng)
    params_to_problem[these_params] = problem_key
    problems[problem_key] = problem
end

#XXX for now just assume that we have lambda and p_normal
#XXX this is not general yet
for p_normal in params["p_normal"]
    problem = problems[Dict("lambda" => params_to_problem[params["lambda"][1]],
                            "p_normal" => p_normal)]
    these_states = assign_keys([initial_state(problem, init_phys_states[i], isrng) for i in 1:N], key_rng)
    for lambda in params["lambda"]
        collect(keys
end
=#

# Calculate single points
point_solvers = Dict{UTF8String, Solver}(
    "random" => RandomSolver(),
    "heuristic" => SimpleSolver()
)

point_results = evaluate(representative_problem, initial_states, point_solvers, parallel=true, N=N)

dpws = DPWSolver(depth=20,
                 n_iterations=300,
                 exploration_constant=50.0,
                 k_state=4.0,
                 alpha_state=1/8,
                 rollout_solver=SimpleSolver()) 

rsolver = RobustMCTSSolver(
    depth=20,
    c=50.0,
    n_iterations=1500,
    k_state=4.0,
    alpha_state=1/8,
    k_nature=4.0,
    alpha_nature=1/8,
    rollout_solver=SimpleSolver(),
    rollout_nature=StochasticBehaviorNoCrashMDP(collect(values(problems))[1]))

pomcps = POMCPDPWSolver(
    eps=0.001,
    max_depth=20,
    c=50.0,
    tree_queries=1500,
    k_observation=4.0,
    alpha_observation=1/8,
    rollout_solver=SimpleSolver()
)

curve_solvers = Dict{UTF8String, Solver}(
    "upper_bound" => dpws,
    "robust" => RobustMLSolver(rsolver),
    "single_behavior" => 
        SingleBehaviorSolver(dpws,
                 IDMMOBILBehavior("normal", 30.0, 10.0, 1)),
    "pomcp" => MLPOMDPSolver(pomcps, BehaviorRootUpdaterStub(0.1))
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

