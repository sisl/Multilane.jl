
#push!(LOAD_PATH,joinpath("..","src"))

using Multilane
using GenerativeModels
using MCTS
using POMDPs
using POMDPToolbox
using Base.Test

#Set up problem configuration
nb_lanes = 4
pp = PhysicalParam(nb_lanes,lane_length=100.) #2.=>col_length=8
_discount = 1.
nb_cars=10

rmodel = NoCrashRewardModel()

dmodel = NoCrashIDMMOBILModel(nb_cars, pp)

mdp = NoCrashMDP(dmodel, rmodel, _discount);

rng = MersenneTwister(5)

s = initial_state(mdp::NoCrashMDP,rng)
#visualize(mdp,s,MLAction(0,0))

policy = RandomPolicy(mdp)

sim = HistoryRecorder(rng=rng, max_steps=100) # initialize a random number generator

simulate(sim, mdp, policy, Multilane.initial_state(mdp, sim.rng))

# check for crashes
for i in 1:length(sim.state_hist)-1
    @test !is_crash(mdp, sim.state_hist[i], sim.action_hist[i])
end
