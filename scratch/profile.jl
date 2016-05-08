using GenerativeModels
using POMDPs
using POMDPToolbox
using ProfileView
using Multilane

nb_lanes = 4
pp = PhysicalParam(nb_lanes,lane_length=100.) #2.=>col_length=8\n",
_discount = 1.
nb_cars=10
rmodel = NoCrashRewardModel()
dmodel = NoCrashIDMMOBILModel(nb_cars, pp)
mdp = NoCrashMDP(dmodel, rmodel, _discount);
rng = MersenneTwister(5)

policy = RandomPolicy(mdp, rng=rng)

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
