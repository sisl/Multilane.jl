using Multilane
using StatsBase

solvers = Dict{UTF8String, Any}(
    "heuristic" => SimpleSolver()
)

curve = TestSet(lambda=[1., 10., 17.78, 31.62, 56.23, 100., 1000.], N=500)

tests = []
push!(tests, TestSet(curve, solver_key="heuristic"))
push!(tests, TestSet(curve, solver_key="heuristic", solver_behavior_probabilities=WeightVec([1.0,0,0])))
# push!(tests, TestSet(curve, solver_key="dpw_300"))
# push!(tests, TestSet(curve, solver_key="dpw_500"))

initials = gen_initials(tests)

@show initials["param_table"]

# results = evaluate(tests, initials, solvers)
