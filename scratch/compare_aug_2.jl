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


problems = Dict{UTF8String, Any}()
key_rng = MersenneTwister(123)

is_rng = MersenneTwister(123)
init_phys_states = [MLPhysicalState(initial_state(base_problem, is_rng)) for i in 1:N]
initial_states = Dict{UTF8String, Any}()
state_lists = Dict{UTF8String, Any}()

params = Dict{Symbol, Any}()
params[:lambda] = Float64[0.1, 1., 10., 17.78, 31.62, 56.23, 100., 1000.]
# params[:lambda] = Float64[0.1, 1.]
params[:p_normal] = Float64[0.2, 1.0]

initial_relevant = [:p_normal]

param_table = DataFrame()
for p in params
    if isempty(param_table)
        param_table = DataFrame(Dict(p))
    else
        param_table = join(param_table, DataFrame(Dict(p)), kind = :cross)
    end
end

param_table[:problem_key] = DataArray(UTF8String, nrow(param_table))
param_table[:state_list_key] = DataArray(UTF8String, nrow(param_table))

for row in eachrow(param_table)
    key = randstring(key_rng)
    problem = deepcopy(base_problem)
    # lambda
    problem.rmodel.cost_dangerous_brake = row[:lambda]*problem.rmodel.reward_in_desired_lane
    # p_normal
    nb_behaviors = length(problem.dmodel.behaviors) 
    probs = ones(nb_behaviors) * (1.0 - row[:p_normal])/(nb_behaviors-1)
    probs[1] = row[:p_normal]
    problem.dmodel.behavior_probabilities = WeightVec(probs)

    problems[key] = problem
    row[:problem_key] = key
end

for g in groupby(param_table, initial_relevant)
    key = randstring(key_rng)
    pk = g[1,:problem_key]
    p = problems[pk]
    these_states = assign_keys([initial_state(p, init_phys_states[i], is_rng) for i in 1:N])
    merge!(initial_states, these_states)
    g[:,:state_list_key] = key
    state_lists[key] = collect(keys(these_states))
end

objects = Dict{UTF8String, Any}()
objects["problems"] = problems
objects["param_table"] = param_table
objects["initial_states"] = initial_states
objects["state_lists"] = state_lists

# Just Figure 1 with 5 simulations

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
    rollout_solver=SimpleSolver())

# for 20% and 100 percent, we need the upper bound, the heuristic and random
point_solvers = Dict{UTF8String, Solver}(
    "random" => RandomSolver(),
    "heuristic" => SimpleSolver()
)

objects["solvers"] = deepcopy(point_solvers)
objects["solvers"]["upper_bound"] = dpws
objects["solvers"]["pomcp"] = MLPOMDPSolver(pomcps, BehaviorRootUpdaterStub(0.1))
objects["solvers"]["robust"] = RobustMLSolver(rsolver)

results = objects

problems_100 = param_table[param_table[:p_normal].==1.0, :problem_key]
problems_20 = param_table[param_table[:p_normal].==0.2, :problem_key]
solver_pairs = [("upper_bound", problems_100),
                ("pomcp", problems_20),
                ("robust", problems_20)]

for p_normal in unique(param_table[:p_normal])
    list = param_table[param_table[:p_normal].==p_normal, :state_list_key][1]
    inits = state_lists[list]
    problems = param_table[param_table[:p_normal].==p_normal, :problem_key]

    r = evaluate([problems[1]], inits, collect(keys(point_solvers)), objects, N=N, desc="$p_normal Point: ")
    merge_results!(results, r)

    r = evaluate(problems, inits, ["upper_bound"], objects, N=N, desc="$p_normal Upper: ")
    merge_results!(results, r)

    for sp in solver_pairs
        r = evaluate(problems, inits, [sp[1]], objects, soln_problem_keys=sp[2], N=N, desc="$p_normal $(sp[1]): ")
        merge_results!(results, r)
    end
end

#=
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
=#

println(results["stats"])

filename = string("results_", Dates.format(Dates.now(),"u_d_HH_MM"), ".jld")
results["histories"] = nothing
save(filename, results)
println("results saved to $filename")

#=
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
=#
