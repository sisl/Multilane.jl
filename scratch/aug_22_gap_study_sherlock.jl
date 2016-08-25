using Multilane
using StatsBase
using MCTS
using JLD


dpws = DPWSolver(depth=20,
                 n_iterations=500,
                 exploration_constant=50.0,
                 k_state=4.0,
                 alpha_state=1/8,
                 rollout_solver=SimpleSolver()) 
solvers = Dict{UTF8String, Any}(
    "dpw"=>dpws,
    "assume_normal"=>SingleBehaviorSolver(dpws, normal)
)

curve = TestSet(lambda=[0.1, 1.0, 2.15, 4.64, 10., 50.,], N=500)

tests = []
for p in linspace(0., 1., 5)
    push!(tests, TestSet(curve, solver_key="dpw",
                         behaviors=@sprintf("agents_%03d", p*100),
                         key=@sprintf("upper_bound_%03d", p*100))) 
    push!(tests, TestSet(curve, solver_key="assume_normal",
                         behaviors=@sprintf("agents_%03d", p*100),
                         key=@sprintf("assume_normal_%03d", p*100))) 
end

objects = gen_initials(tests, behaviors=behaviors, generate_physical=true)

@show objects["param_table"] 
objects["solvers"] = solvers
for is in values(objects["initial_states"])
    @assert !isnull(is.env_cars[1].behavior)
end

results = evaluate(tests, objects, parallel=true)

filename = string("results_", Dates.format(Dates.now(),"u_d_HH_MM"), ".jld")
results["histories"] = nothing
save(filename, results)
println("results saved to $filename")
