
#push!(LOAD_PATH,joinpath("..","src"))

#using Multilane
#using GenerativeModels
#using MCTS
#using POMDPs
#using POMDPToolbox

#Set up problem configuration
nb_lanes = 4
pp = PhysicalParam(nb_lanes,lane_length=100.) #2.=>col_length=8
_discount = 1.
nb_cars=10

rmodel = NoCrashRewardModel()

dmodel = NoCrashIDMMOBILModel(nb_cars, pp)

mdp = NoCrashMDP(dmodel, rmodel, _discount);

rng = MersenneTwister(5)

s = Multilane.create_state(mdp)
s = Multilane.initial_state(mdp::NoCrashMDP,rng,s)
#visualize(mdp,s,MLAction(0,0))

policy = RandomPolicy(mdp)

sim = HistoryRecorder(rng=MersenneTwister(9), max_steps=10) # initialize a random number generator

simulate(sim, mdp, policy, Multilane.initial_state(mdp, sim.rng))
