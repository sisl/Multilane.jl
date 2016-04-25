using GenerativeModels
using POMDPs
using POMDPToolbox
using ProfileView
using Multilane

#Set up problem configuration
nb_lanes = 2
pp = PhysicalParam(nb_lanes,lane_length=200.) #2.=>col_length=8
r_crash = -1.
accel_cost = -1e-5
decel_cost = -5e-6
invalid_cost = -1e-5
lineride_cost = -1e-5
lanechange_cost = -2e-5
_discount = 1.
nb_cars=3

rng = MersenneTwister(9)

rmodel = OriginalRewardModel(r_crash,
                             accel_cost,
                             decel_cost,
                             invalid_cost,
                             lineride_cost,
                             lanechange_cost)

dmodel = IDMMOBILModel(nb_cars, pp)

mdp = OriginalMDP(dmodel, rmodel, _discount)

policy = RandomPolicy(mdp)

sim = RolloutSimulator(max_steps=1000)

@time simulate(sim, mdp, policy, initial_state(mdp, rng))
@time for i in 1:100
    simulate(sim, mdp, policy, initial_state(mdp, rng))
end

Profile.clear()
@profile for i in 1:100
    simulate(sim, mdp, policy, initial_state(mdp, rng))
end

ProfileView.view()
