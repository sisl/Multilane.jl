using Multilane
using StatsBase
using MCTS
using JLD
using POMCP

behaviors = Dict{UTF8String,Any}(
    "uniform" => standard_uniform(1.0, correlated=false)
)

dpws = DPWSolver(depth=20,
                 n_iterations=500,
                 exploration_constant=50.0,
                 k_state=4.0,
                 alpha_state=1/8,
                 rollout_solver=SimpleSolver()) 

pomcps = POMCPDPWSolver(
    eps=0.001,
    max_depth=20,
    c=50.0,
    tree_queries=2500,
    k_observation=4.0,
    alpha_observation=1/8,
    rollout_solver=SimpleSolver()
)

# agg_up = AggressivenessUpdater(nothing, 500, 0.1, 0.1, WeightUpdateParams(smoothing=0.0, wrong_lane_factor=0.5), MersenneTwister(123))
up = BehaviorParticleUpdater(nothing, 1000, 0.1, 0.2, WeightUpdateParams(smoothing=0.0, wrong_lane_factor=0.5), MersenneTwister(123))


solvers = Dict{UTF8String, Any}(
    "dpw"=>dpws,
    "assume_normal"=>SingleBehaviorSolver(dpws, Multilane.NORMAL),
    "pomcp"=>MLPOMDPSolver(pomcps, up),
    "mlmpc"=>MLMPCSolver(dpws, up)
)

curve = TestSet(lambda=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0], p_appear=1.0, brake_threshold=4.0, N=500, behaviors="uniform")

tests = [
    TestSet(curve, solver_key="dpw", key="upper_bound_unif"),
    TestSet(curve, solver_key="assume_normal", key="assume_normal_unif"),
    TestSet(curve, solver_key="pomcp", key="pomcp_unif"),
    TestSet(curve, solver_key="mlmpc", key="mlmpc_unif")
]

objects = gen_initials(tests, behaviors=behaviors, generate_physical=true)

@show objects["param_table"] 
objects["solvers"] = solvers
objects["note"] = "pomcp and mlmpc with a "

files = sbatch_spawn(tests, objects,
                     batch_size=50,
                     time_per_batch="1:00:00",
                     submit_command="sbatch",
                     template_name="sherlock.sh")

# files = sbatch_spawn(tests, objects,
#                      batch_size=6,
#                      time_per_batch="1:00:00",
#                      submit_command="bash",
#                      template_name="theresa.sh")

# results = evaluate(tests, objects, parallel=true)
# 
# filename = string("results_", Dates.format(Dates.now(),"u_d_HH_MM"), ".jld")
# results["histories"] = nothing
# save(filename, results)
# println("results saved to $filename")
