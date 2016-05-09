procs = addprocs(7)

using Multilane
using POMDPToolbox
using GenerativeModels
using MCTS


nb_lanes = 4
pp = PhysicalParam(nb_lanes,lane_length=100.) #2.=>col_length=8\n",
_discount = 1.
nb_cars=10
rmodel = NoCrashRewardModel(100,10,1.5,nb_lanes)
dmodel = NoCrashIDMMOBILModel(nb_cars, pp)
mdp = NoCrashMDP(dmodel, rmodel, _discount);
isrng = MersenneTwister(123)

N = 100

problems = [mdp for i in 1:N]
initial_states = [initial_state(mdp, isrng) for i in 1:N]

rs = RandomSolver()

r_random = Multilane.evaluate_performance(problems, initial_states, rs)

dpws = DPWSolver()

r_dpw = Multilane.evaluate_performance(problems, initial_states, dpws, parallel=false)

single_dpws = SingleBehaviorSolver(dpws, IDMMOBILBehavior("normal", 30.0, 10.0, 1))

r_single_dpw = Multilane.evaluate_performance(problems, initial_states, single_dpws)

println("Random Average Reward: $(mean(r_random))")
println("DPW Average Reward: $(mean(r_dpw))")
println("Single Behavior DPW Average Reward: $(mean(r_single_dpw))")

rmprocs(procs)
