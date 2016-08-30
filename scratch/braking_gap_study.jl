using Multilane
using StatsBase
using MCTS
using JLD

behaviors=Dict{UTF8String,Any}(
    "uniform" => standard_uniform()
)

dpws = DPWSolver(depth=20,
                 n_iterations=500,
                 exploration_constant=50.0,
                 k_state=4.0,
                 alpha_state=1/8,
                 rollout_solver=SimpleSolver()) 

solvers = Dict{UTF8String, Any}(
    "dpw"=>dpws,
    "assume_normal"=>SingleBehaviorSolver(dpws, Multilane.NORMAL)
)

curve = TestSet(lambda=[1.0, 1.78, 3.16, 5.62, 10., 31.6], N=500, p_appear=1.0)

tests = []
for b in [2.0, 3.0, 3.5, 4.0]
    push!(tests, TestSet(curve, solver_key="dpw",
                         behaviors="uniform",
                         key=@sprintf("upper_bound_%02d", b*10),
                         brake_threshold=b)) 
    push!(tests, TestSet(curve, solver_key="assume_normal",
                         behaviors="uniform",
                         key=@sprintf("assume_normal_%02d", b*10),
                         brake_threshold=b)) 
end

objects = gen_initials(tests, behaviors=behaviors, generate_physical=true)

@show objects["param_table"] 
objects["solvers"] = solvers
for is in values(objects["initial_states"])
    @assert !isnull(is.env_cars[1].behavior)
end

objects["note"] = "Tests over a range of braking values."

sbatch_spawn(tests, objects,
             batch_size=50,
             time_per_batch="1:00:00",
             submit_command="sbatch",
             template_name="sherlock.sh")

# sbatch_spawn(tests, objects,
#              batch_size=12,
#              time_per_batch="1:00:00",
#              submit_command="bash",
#              template_name="theresa.sh")


# results = evaluate(tests, objects, parallel=true)
# 
# filename = string("results_", Dates.format(Dates.now(),"u_d_HH_MM"), ".jld")
# results["histories"] = nothing
# save(filename, results)
# println("results saved to $filename")
