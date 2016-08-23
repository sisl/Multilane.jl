using Multilane
using StatsBase
using MCTS
using JLD
# using Plots
# using StatPlots

solvers = Dict{UTF8String, Any}(
    "normal" => BehaviorSolver(Multilane.NORMAL, false, MersenneTwister(123)),
    "lane_seeking" => IDMLaneSeekingSolver(Multilane.NORMAL, MersenneTwister(123))
)

test = TestSet(lambda=10., solver_key="lane_seeking", behaviors="agents", N=500)

objects = gen_initials([test])

@show objects["param_table"] 
objects["solvers"] = solvers

metrics=[MaxBrakeMetric(), NumBehaviorBrakesMetric(3,"aggressive",2.5)]
objects["metrics"] = metrics
# results = evaluate([test], objects, metrics=metrics, max_steps=200)
results_file_list = sbatch_spawn([test], objects,
                                 submit_command="sbatch",
                                 template_name="sherlock.sh",
                                 time_per_batch="10:00")

#=

results = gather_results(results_file_list)
stats = results["stats"]
@show names(results["stats"])

@show sum(results["stats"][:aggressive_nb_brakes])/sum(results["stats"][:nb_brakes])


# unicodeplots()
histogram(results["stats"][:max_brake], title="max brake histogram")
gui()

i = 20
problem, sim, policy = rerun(results, i);
f = write_tmp_gif(problem, sim)
run(`gifview $f`)

=#

# filename = string("results_", Dates.format(Dates.now(),"u_d_HH_MM"), ".jld")
# results["histories"] = nothing
# save(filename, results)
# println("results saved to $filename")
