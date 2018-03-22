using Multilane
using StatsBase

solvers = Dict{String, Any}()

dpws = DPWSolver(depth=20,
                 n_iterations=100,
                 exploration_constant=50.0,
                 k_state=4.0,
                 alpha_state=1/8,
                 rollout_solver=SimpleSolver()) 

solvers["dpw_100"] = deepcopy(dpws)
dpws.n_iterations = 300
solvers["dpw_300"] = deepcopy(dpws)
dpws.n_iterations = 500
solvers["dpw_500"] = deepcopy(dpws)

curve = TestSet(lambda=[1., 10., 17.78, 31.62, 56.23, 100.], N=500)

tests = []
for (s,_) in solvers
    push!(tests, TestSet(curve, solver_key=s)) 
end

objects = gen_initials(tests)

@show objects["param_table"] 
objects["solvers"] = solvers

results = evaluate(tests, objects)
