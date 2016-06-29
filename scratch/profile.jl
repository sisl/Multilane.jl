using GenerativeModels
using POMDPs
using POMDPToolbox
using ProfileView
using Multilane
using MCTS

nb_lanes = 4
pp = PhysicalParam(nb_lanes,lane_length=100.) #2.=>col_length=8\n",
_discount = 1.
nb_cars=10
rmodel = NoCrashRewardModel()
dmodel = NoCrashIDMMOBILModel(nb_cars, pp)
mdp = NoCrashMDP(dmodel, rmodel, _discount);
rng = MersenneTwister(5)


dpws = DPWSolver(depth=20,
                 n_iterations=1000,
                 exploration_constant=100.0,
                 k_state=4.0,
                 alpha_state=1/8,
                 rollout_solver=SimpleSolver()) 

policy = solve(dpws, mdp)

# policy = RandomPolicy(mdp, rng=rng)

sim = RolloutSimulator(max_steps=100)

@time simulate(sim, mdp, policy, initial_state(mdp, rng))
@time for i in 1
    simulate(sim, mdp, policy, initial_state(mdp, rng))
end

Profile.clear()
@profile for i in 1
    simulate(sim, mdp, policy, initial_state(mdp, rng))
end

ProfileView.view()
