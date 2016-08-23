using Multilane
using StatsBase
using MCTS
using JLD
using POMCP

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


solvers = Dict{UTF8String, Any}(
    "dpw"=>dpws,
    "assume_normal"=>SingleBehaviorSolver(dpws, Multilane.NORMAL),
    "pomcp"=>MLPOMDPSolver(pomcps, BehaviorRootUpdaterStub(0.05))
)

# curve = TestSet(lambda=[0.1, 1.0, 2.15, 4.64, 10., 21.5, 46.4], N=1)
curve = TestSet(lambda=[46.4], N=1)

tests = [
    TestSet(curve, solver_key="dpw", behaviors="agents"),
    TestSet(curve, solver_key="assume_normal", behaviors="agents"),
    TestSet(curve, solver_key="pomcp", behaviors="agents")
]

objects = gen_initials(tests, generate_physical=true)

@show objects["param_table"] 
objects["solvers"] = solvers

files = sbatch_spawn(tests, objects,
                     batch_size=1,
                     time_per_batch="10:00",
                     submit_command="sbatch",
                     template_name="sherlock")

# results = evaluate(tests, objects, parallel=true)

# filename = string("results_", Dates.format(Dates.now(),"u_d_HH_MM"), ".jld")
# results["histories"] = nothing
# save(filename, results)
# println("results saved to $filename")
