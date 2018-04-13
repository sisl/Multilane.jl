using Multilane
using POMDPs
using POMDPToolbox
using BenchmarkTools

nb_lanes = 4
pp = PhysicalParam(nb_lanes,lane_length=200.) #2.=>col_length=8
_discount = 1.
nb_cars=10

rmodel = NoCrashRewardModel()

dmodel = NoCrashIDMMOBILModel(nb_cars, pp)

mdp = NoCrashMDP{typeof(rmodel), typeof(dmodel.behaviors)}(dmodel, rmodel, _discount, true);

rng = MersenneTwister(5)

s = initial_state(mdp::NoCrashMDP, rng)

policy = Multilane.BehaviorPolicy(mdp, Multilane.NORMAL, false, rng)

hr = HistoryRecorder(rng=rng, max_steps=10000)

hist = simulate(hr, mdp, policy, s)
@time hist = simulate(hr, mdp, policy, s)
@show n_steps(hist)
hrbm = @benchmark simulate(hr, mdp, policy, s)
@show hrbm

ro = RolloutSimulator(rng=rng, max_steps=10000)
simulate(ro, mdp, policy, s)
@time simulate(ro, mdp, policy, s)
robm = @benchmark simulate(ro, mdp, policy, s)
@show robm
