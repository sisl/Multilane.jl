using Multilane
using StatsBase
using MCTS
using JLD

behaviors=Dict{String,Any}()
for p in linspace(0., 1.25, 6)
    behaviors[@sprintf("uniform_%03d", 100*p)] = standard_uniform(p)
end

dpws = DPWSolver(depth=20,
                 n_iterations=500,
                 exploration_constant=50.0,
                 k_state=4.0,
                 alpha_state=1/8,
                 rollout_solver=SimpleSolver()) 
solvers = Dict{String, Any}(
    "dpw"=>dpws,
    "assume_normal"=>SingleBehaviorSolver(dpws, Multilane.NORMAL)
)

curve = TestSet(lambda=[0.1, 1.0, 2.15, 4.64, 10., 50.,], N=500, p_appear=1.0)

tests = []
for p in linspace(0., 1.25, 6)
    push!(tests, TestSet(curve, solver_key="dpw",
                         behaviors=@sprintf("uniform_%03d", p*100),
                         key=@sprintf("upper_bound_%03d", p*100))) 
    push!(tests, TestSet(curve, solver_key="assume_normal",
                         behaviors=@sprintf("uniform_%03d", p*100),
                         key=@sprintf("assume_normal_%03d", p*100))) 
end

objects = gen_initials(tests, behaviors=behaviors, generate_physical=true)

@show objects["param_table"] 
objects["solvers"] = solvers
for is in values(objects["initial_states"])
    @assert !isnull(is.cars[1].behavior)
end

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
