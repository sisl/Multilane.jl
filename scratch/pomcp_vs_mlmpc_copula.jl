using Multilane
using StatsBase
using MCTS
using JLD
using POMCP

cors = [0.5, 0.75, 0.9]

behaviors = Dict{UTF8String,Any}()
for cor in cors
    behaviors[@sprintf("cor_%03d", 100*cor)] = standard_uniform(1.0, correlation=cor)
end

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

agg_up = AggressivenessUpdater(nothing, 500, 0.1, 0.1, WeightUpdateParams(smoothing=0.0, wrong_lane_factor=0.5), MersenneTwister(123))

solvers = Dict{UTF8String, Any}(
    "dpw"=>dpws,
    "assume_normal"=>SingleBehaviorSolver(dpws, Multilane.NORMAL),
    "pomcp"=>MLPOMDPSolver(pomcps, agg_up),
    "mlmpc"=>MLMPCSolver(dpws, agg_up)
)

curve = TestSet(lambda=[1.0, 2.0, 4.0, 8.0, 16.0, 32.0], p_appear=1.0, brake_threshold=4.0, N=500)
# curve = TestSet(lambda=[46.4], N=1)

tests = []
for cor in cors
    bg = @sprintf("cor_%03d", 100*cor)
    push!(tests, TestSet(curve, solver_key="dpw", key=@sprintf("upper_bound_%03d", 100*cor), behaviors=bg))
    push!(tests, TestSet(curve, solver_key="assume_normal", key=@sprintf("assume_normal_%03d", 100*cor), behaviors=bg))
    push!(tests, TestSet(curve, solver_key="pomcp", key=@sprintf("pomcp_%03d", 100*cor), behaviors=bg))
    push!(tests, TestSet(curve, solver_key="mlmpc", key=@sprintf("mlmpc_%03d", 100*cor), behaviors=bg))
end

objects = gen_initials(tests, behaviors=behaviors, generate_physical=true)

@show objects["param_table"] 
objects["solvers"] = solvers
objects["note"] = "pomcp vs mlmpc with various levels of correlation"

files = sbatch_spawn(tests, objects,
                     batch_size=50,
                     time_per_batch="1:00:00",
                     submit_command="sbatch",
                     template_name="sherlock.sh")

# files = sbatch_spawn(tests, objects,
#                      batch_size=12,
#                      time_per_batch="1:00:00",
#                      submit_command="bash",
#                      template_name="theresa.sh")

# results = evaluate(tests, objects, parallel=true)
# 
# filename = string("results_", Dates.format(Dates.now(),"u_d_HH_MM"), ".jld")
# results["histories"] = nothing
# save(filename, results)
# println("results saved to $filename")
