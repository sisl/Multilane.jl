procs = addprocs(2)


push!(LOAD_PATH,joinpath("..","src"))
using Multilane
using POMCP
using POMDPToolbox
using GenerativeModels
using MCTS
using JLD


function write!(var_name::AbstractString, val::NoCrashStats, fname::AbstractString="results.jld")
  # XXX kinda hacky but AFAIK JLD has no append functionality
  # NOTE im not sure how to use this functionality AND do the addrequire stuff
  if !isfile(fname)
    d = Dict{ByteString,Any}()
  else
    d = load(fname)
  end
  if var_name in keys(d)
    warn("trying to overwrite value, adding spaghetti")
    var_name = string(var_name,"99999")
  end
  d[var_name] = val
  save(fname, d)
end

function test_solver(problems::Array{NoCrashMDP,1}, solver, solver_name::AbstractString, initial_states::Array{MLState,1})
  r = Multilane.evaluate_performance(problems, initial_states, solver)
  write!(solver_name, r)
  return r
end

function test_solver(problems::Array{NoCrashPOMDP,1}, solver, solver_name::AbstractString, initial_states::Array{MLState,1}, bu)
  r = Multilane.evaluate_performance(problems, initial_states, solver, bu)
  write!(solver_name, r)
  return r
end

function test_pomdp(problem::NoCrashPOMDP, solver, solver_name::AbstractString, initial_states, bu)
  r = NoCrashStats([test_run(problem, bu, s, solver) for s in initial_states], problem.rmodel)
  write!(solver_name, r)
  return r
end

nb_lanes = 4
pp = PhysicalParam(nb_lanes,lane_length=100.) #2.=>col_length=8\n",
_discount = 1.
nb_cars=10
dmodel = NoCrashIDMMOBILModel(nb_cars, pp)

N = 10

models = JLD.load("rmodels.jld")
rmodels = models["rmodels"]

# NOTE: random solver and heuristic solver are rmodel agnostic--won't get pareto curve
rmodel = rmodels[1]
mdp = NoCrashMDP(dmodel, rmodel, _discount);
isrng = MersenneTwister(123)
# Initial state is indifferent to rmodel
initial_states = MLState[initial_state(mdp, isrng) for i in 1:N]
problems = NoCrashMDP[mdp for i in 1:N]

r_random = test_solver(problems, RandomSolver(), "random", initial_states)

println("Random Average Reward: $(mean(r_random))")

r_heur = test_solver(problems, SimpleSolver(), "simple_heuristic", initial_states)

println("Heuristic Average Reward: $(mean(r_heur))")

pomdp = NoCrashPOMDP(dmodel, rmodel, _discount)

problems = NoCrashPOMDP[pomdp for i in 1:N]

bu = ParticleUpdater(100, pomdp, MersenneTwister(555))

r_pomcp = test_pomdp(pomdp, POMCPSolver(updater=bu), string("pomcp", 0), initial_states, bu)

println("POMCP Avg Reward: $(mean(r_pomcp))")


for (k,rmodel) in enumerate(rmodels[1:2])
  mdp = NoCrashMDP(dmodel, rmodel, _discount);

  problems = NoCrashMDP[mdp for i in 1:N]

  dpws = DPWSolver()

  r_dpw = test_solver(problems, dpws, string("dpw", k), initial_states)

  println("DPW Average Reward: $(mean(r_dpw))")
  single_dpws = SingleBehaviorSolver(dpws, IDMMOBILBehavior("normal", 30.0, 10.0, 1))

  r_single_dpw = test_solver(problems, single_dpws, string("single_dpw", k), initial_states)

  println("Single Behavior DPW Average Reward: $(mean(r_single_dpw))")


  # POMDP stuff

  #=
  pomdp = NoCrashPOMDP(dmodel, rmodel, _discount)

  problems = NoCrashPOMDP[pomdp for i in 1:N]

  bu = ParticleUpdater(100, pomdp, MersenneTwister(555))

  r_pomcp = test_pomdp(pomdp, POMCPSolver(updater=bu), string("pomcp", k), initial_states, bu)

  println("POMCP Avg Reward: $(mean(r_pomcp))")
  =#

  #r_despot = test_solver(problems, DespotSolver(pomdp,belief), string("dpw", k), initial_states)

  #println("DESPOT Avg Reward: $(mean(r_despot))")

end


rmprocs(procs)
