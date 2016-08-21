using Multilane
using StatsBase
using MCTS
using JLD


# from "Agents" paper
# bsafe = b
# a_thr = 0.1 # from the "General Lane-Changing" paper
# p in [0.2, 1] from Agents Paper, set to 0.5 for all
normal_idm = IDMParam(1.4, 2.0, 1.5, 120/3.6, 2.0, 4.0) 
timid_idm = IDMParam(1.0, 1.0, 1.8, 100/3.6, 4.0, 4.0)
aggressive_idm = IDMParam(2.0, 3.0, 1.0, 140/3.6, 1.0, 4.0)
agents_behavior(idm, idx) = IDMMOBILBehavior(idm, MOBILParam(0.5, idm.b, 0.1), idx)

normal = agents_behavior(normal_idm, 1)
timid = agents_behavior(timid_idm, 2)
aggressive = agents_behavior(aggressive_idm, 3)

behaviors = Dict{UTF8String, Any}(
    "all_normal" => DiscreteBehaviorSet([normal], WeightVec([1])),
)

for p in linspace(0., 1., 5)
    behaviors[@sprintf("agents_%03d", 100*p)] =
            DiscreteBehaviorSet([normal, timid, aggressive],
                                WeightVec([p, (1-p)/2, (1-p)/2]))
end

# @show behaviors

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

curve = TestSet(lambda=[1.0, 2.15, 4.64, 10., 21.5, 46.4, 100.,], N=500)

tests = []
for p in linspace(0., 1., 5)
    push!(tests, TestSet(curve, solver_key="dpw",
                         behaviors=@sprintf("agents_%03d", p*100),
                         key=@sprintf("upper_bound_%03d", p*100))) 
    push!(tests, TestSet(curve, solver_key="assume_normal",
                         behaviors=@sprintf("agents_%03d", p*100),
                         key=@sprintf("assume_normal_%03d", p*100))) 
end

objects = gen_initials(tests, behaviors=behaviors)

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
