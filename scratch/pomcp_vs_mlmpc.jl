using Multilane
using StatsBase
using MCTS
using JLD
using POMCP

behaviors = Dict{UTF8String,Any}(
    "correlated" => standard_uniform(1.0, correlated=true)
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

agg_up = AggressivenessUpdater(nothing, 100, 0.015, WeightUpdateParams(smoothing=0.0), MersenneTwister(123))


solvers = Dict{UTF8String, Any}(
    "dpw"=>dpws,
    "assume_normal"=>SingleBehaviorSolver(dpws, Multilane.NORMAL),
    "pomcp"=>MLPOMDPSolver(pomcps, agg_up),
    "mlmpc"=>MLMPCSolver(dpws, agg_up)
)

curve = TestSet(lambda=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0], p_appear=1.0, brake_threshold=2.5, N=500, behaviors="correlated")
# curve = TestSet(lambda=[46.4], N=1)

tests = [
    TestSet(curve, solver_key="dpw", key="dpw"),
    TestSet(curve, solver_key="assume_normal", key="assume_normal"),
    TestSet(curve, solver_key="pomcp", key="pomcp"),
    TestSet(curve, solver_key="mlmpc", key="mlmpc")
]

objects = gen_initials(tests, behaviors=behaviors, generate_physical=true)

@show objects["param_table"] 
objects["solvers"] = solvers

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
