using Multilane
using StatsBase
using MCTS
using JLD
using POMDPToolbox
using POMDPs
# using POMCP

using Gallium

dpws = DPWSolver(depth=20,
                 # n_iterations=500,
                 max_time=1.0,
                 exploration_constant=50.0,
                 k_state=4.0,
                 alpha_state=1/8,
                 estimate_value=RolloutEstimator(SimpleSolver()))


solver = SingleBehaviorSolver(dpws, Multilane.NORMAL)


lambda = 10.0
behaviors = standard_uniform(correlation=0.75)
pp = PhysicalParam(4, lane_length=100.0)
dmodel = NoCrashIDMMOBILModel(10, pp,
                              behaviors=behaviors,
                              p_appear=1.0,
                              lane_terminate=false)
rmodel = NoCrashRewardModel()
rmodel.brake_penalty_thresh = 4.0
rmodel.cost_dangerous_brake = lambda*rmodel.reward_in_target_lane
pomdp = NoCrashPOMDP{typeof(rmodel)}(dmodel, rmodel, 0.95)
mdp = NoCrashMDP{typeof(rmodel)}(dmodel, rmodel, 0.95)
is = initial_state(pomdp, Base.GLOBAL_RNG)
ips = MLPhysicalState(is)

planner = solve(solver, mdp)
agg_up = AggressivenessUpdater(pomdp, 500, 0.1, 0.1, WeightUpdateParams(smoothing=0.0, wrong_lane_factor=0.5), MersenneTwister(123))


for 
    sims = [Sim(pomdp, planner, agg_up, ips, is, max_steps=100)]

end
