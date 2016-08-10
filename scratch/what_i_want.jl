using Multilane
using StatsBase

solvers = Dict{UTF8String, Any}(
    "heuristic" => SimpleSolver()
)

curve = TestSet(lambda=[1., 10., 17.78, 31.62, 56.23, 100., 1000.], N=1)

tests = []
push!(tests, TestSet(curve, solver_key="heuristic"))
push!(tests, TestSet(curve, solver_key="heuristic", solver_behaviors="3_even"))

objects = gen_initials(tests)

@show objects["param_table"] 
objects["solvers"] = solvers

results = evaluate(tests, objects)
