procs = addprocs(1)


push!(LOAD_PATH,joinpath("..","src"))
push!(LOAD_PATH,joinpath("..","..","..","POMCP.jl","src"))
using Multilane
using POMCP
using POMDPToolbox
using GenerativeModels
using MCTS
using JLD


function POMCP.extract_belief(::POMDPToolbox.FastPreviousObservationUpdater{MLObs}, node::RootNode)
  rand(MersenneTwister(1),node.B)
end

POMCP.initialize_belief(u::FastPreviousObservationUpdater{MLObs}, o::Union{MLState,MLObs}) = o

POMCP.create_belief(u::FastPreviousObservationUpdater{MLObs}) = nothing

POMCP.extract_belief(::POMDPToolbox.FastPreviousObservationUpdater{MLObs}, node::BeliefNode) = node.label[2]


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
  # TODO addrequire
  save(fname, d)
end

function test_solver(problems::Array{NoCrashMDP,1}, solver, solver_name::AbstractString, initial_states::Array{MLState,1})
  if typeof(solver) <: DPWSolver
    _ec = solver.exploration_constant
    best_score = -Inf
    for (i,ec) in enumerate(Float64[0.1,0.3,1.,3.,10.,30.,100.])
      solver.exploration_constant = ec
      _r = Multilane.evaluate_performance(problems, initial_states, solver)
      score = mean(_r)[3]
      if score > best_score
        r = _r
        best_score = score
      end
    end
    solver.exploration_constant = _ec
  else
    r = Multilane.evaluate_performance(problems, initial_states, solver)
  end
  # TODO hyperparameter optimization here, save best
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

#r_random = test_solver(problems, RandomSolver(), "random", initial_states)

#println("Random Average Reward: $(mean(r_random))")

#r_heur = test_solver(problems, SimpleSolver(), "simple_heuristic", initial_states)

#println("Heuristic Average Reward: $(mean(r_heur))")

# exploration constants for POMCP
ecs = [3., 5., 20., 50., 150., 300., 600., 1500., 3000., 5000.]


for (k,(rmodel,ec)) in enumerate(zip(rmodels,ecs))
  mdp = NoCrashMDP(dmodel, rmodel, _discount);

  problems = NoCrashMDP[mdp for i in 1:N]

  dpws = DPWSolver(depth=10,n_iterations=100) #k_action, alpha_action, k_state, alpha_state

  #r_dpw = test_solver(problems, dpws, string("dpw", k), initial_states)

  #println("DPW Average Reward: $(mean(r_dpw))")

  single_dpws = SingleBehaviorSolver(dpws, IDMMOBILBehavior("normal", 30.0, 10.0, 1))

  #r_single_dpw = test_solver(problems, single_dpws, string("single_dpw", k), initial_states)

  #println("Single Behavior DPW Average Reward: $(mean(r_single_dpw))")


  # POMDP stuff

  pomdp = NoCrashPOMDP(dmodel, rmodel, _discount)

  problems = NoCrashPOMDP[pomdp for i in 1:N]

  bu = ParticleUpdater(100, pomdp, MersenneTwister(555))

  pomcp = POMCPDPWSolver( k_observation=0.3,
                          alpha_observation=0.75,
                          k_action=0.5,
                          alpha_action=5.,
                          tree_queries=100,
                          c=ec,
                          eps=0.01)
                          #rollout_solver=SimpleSolver())

                          # Can't use simple solver! Issue is that somehow POMCPDPW is somehow generating an invalid action...

  #pomcp = POMCPSolver()
  r_pomcp = test_pomdp(pomdp, pomcp, string("pomcp", k), initial_states, bu)

  println("POMCP Avg Reward: $(mean(r_pomcp))")

  #r_despot = test_solver(problems, DespotSolver(pomdp,belief), string("dpw", k), initial_states)

  #println("DESPOT Avg Reward: $(mean(r_despot))")

end


rmprocs(procs)
