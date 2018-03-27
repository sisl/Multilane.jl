using ProfileView
using MCTS
using POMDPToolbox
using POMDPs
using Multilane
using POMCPOW
using D3Trees

tii = true
wup = WeightUpdateParams(smoothing=0.0, wrong_lane_factor=0.5)

# solver = POMCPOWSolver(tree_queries=10_000_000,
#                                criterion=MaxUCB(10.0),
#                                max_depth=20,
#                                max_time=1.0,
#                                enable_action_pw=false,
#                                k_observation=4.0,
#                                alpha_observation=1/8,
#                                estimate_value=FORollout(SimpleSolver()),
#                                check_repeat_obs=false,
#                                node_sr_belief_updater=AggressivenessPOWFilter(wup),
#                                tree_in_info=tii
#                               )

dpws = DPWSolver(depth=20,
                 n_iterations=20_000,
                 max_time=1.0,
                 exploration_constant=10.0,
                 k_state=4.0,
                 alpha_state=1/8,
                 enable_action_pw=false,
                 check_repeat_state=false,
                 estimate_value=RolloutEstimator(SimpleSolver()),
                 tree_in_info=tii
                )
# solver = MLMPCSolver(dpws)

solver = GenQMDPSolver(dpws)

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

qmdp = QMDPWrapper(mdp, typeof(ib))
# planner = solve(solver, pomdp)
planner = solve(solver, qmdp)

@show collect(actions(pomdp, ips))
@show collect(actions(pomdp, ib))
@show collect(actions(qmdp, state(qmdp, is)))
@show collect(actions(qmdp, state(qmdp, ib)))

action(planner, ib)
@time action(planner, ib)
Profile.clear()
@profile action(planner, ib)

a, i = action_info(planner, ib)
@show i[:tree_queries]

ProfileView.view()
inchrome(D3Tree(i[:tree]))
