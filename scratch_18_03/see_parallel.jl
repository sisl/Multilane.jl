using MCTS
using POMDPToolbox
using POMDPs
using Multilane
using POMCPOW
using D3Trees
using ParallelPOMCPOW

@everywhere begin
    using POMDPs
    using ParallelPOMCPOW
    using Multilane
    POMDPs.actions(p::MLPOMDP, fsrb::ParallelPOMCPOW.FilteredSRBelief) = actions(p, fsrb.srb)
end

tii = true
wup = WeightUpdateParams(smoothing=0.0, wrong_lane_factor=0.5)

solver = POMCPOWSolver(tree_queries=10_000,
                               criterion=MaxUCB(10.0),
                               max_depth=20,
                               max_time=Inf,
                               enable_action_pw=false,
                               k_observation=4.0,
                               alpha_observation=1/8,
                               estimate_value=FORollout(SimpleSolver()),
                               check_repeat_obs=false,
                               node_sr_belief_updater=AggressivenessPOWFilter(wup),
                               tree_in_info=tii
                              )

psolver = ParallelPOMCPOWSolver(solver, 1000)

behaviors = standard_uniform(correlation=0.75)
pp = PhysicalParam(4, lane_length=100.0)
dmodel = NoCrashIDMMOBILModel(10, pp,
                              behaviors=behaviors,
                              p_appear=1.0,
                              lane_terminate=true,
                              max_dist=500.0
                             )
rmodel = SuccessReward(lambda=2,
                       target_lane=4,
                       brake_penalty_thresh=4.0
                      )
pomdp = NoCrashPOMDP{typeof(rmodel)}(dmodel, rmodel, 0.95, false)
mdp = NoCrashMDP{typeof(rmodel)}(dmodel, rmodel, 0.95, false)
is = initial_state(pomdp, Base.GLOBAL_RNG)
ips = MLPhysicalState(is)

agg_up = AggressivenessUpdater(pomdp, 500, 0.1, 0.1, wup, MersenneTwister(50000))
ib = initialize_belief(agg_up, ips)

planner = solve(solver, pomdp)
pplanner = solve(psolver, pomdp)

println("Single Thread")
action(planner, ib)
@time action(planner, ib)

println("Parallel")
action(pplanner, ib)
@time action(pplanner, ib)
