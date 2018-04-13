using POMDPToolbox
using POMDPs
using POMCPOW
using Multilane
using AutoViz
using Reel
using ProgressMeter
using AutomotiveDrivingModels
using ImageView
using Images
using D3Trees

behaviors = standard_uniform(correlation=0.75)
pp = PhysicalParam(4, lane_length=120.0)
dmodel = NoCrashIDMMOBILModel(10, pp,
                              behaviors=behaviors,
                              p_appear=1.0,
                              lane_terminate=true,
                              max_dist=1000.0
                             )
rmodel = SuccessReward(lambda=1.0, speed_thresh=20.0, lane_change_cost=0.05)
pomdp = NoCrashPOMDP{typeof(rmodel), typeof(dmodel.behaviors)}(dmodel, rmodel, 0.95, true);

if !isdefined(:hist) || hist == nothing
    rng = MersenneTwister(11)

    s = initial_state(pomdp, rng)

    @show n_iters = 10000
    @show max_time = Inf
    @show max_depth = 40
    @show val = SimpleSolver()

    wup = WeightUpdateParams(smoothing=0.0, wrong_lane_factor=0.5)
    solver = POMCPOWSolver(tree_queries=n_iters,
                           criterion=MaxUCB(2.0),
                           max_depth=max_depth,
                           max_time=max_time,
                           enable_action_pw=false,
                           k_observation=4.0,
                           alpha_observation=1/20.0,
                           estimate_value=FORollout(val),
                           # estimate_value=val,
                           check_repeat_obs=false,
                           node_sr_belief_updater=AggressivenessPOWFilter(wup),
                           rng=MersenneTwister(7),
                           tree_in_info=true
                          )
    planner = solve(solver, pomdp)

    sim = HistoryRecorder(rng=rng, max_steps=2, show_progress=true) # initialize a random number generator

    up = BehaviorParticleUpdater(pomdp, 1000, 0.1, 0.1, wup, MersenneTwister(50000))

    hist = simulate(sim, pomdp, planner, up, MLPhysicalState(s), s)
end

@show discounted_reward(hist)

s, ai, r, sp = first(eachstep(hist, "s, ai, r, sp"))
t = 0.0
tree = ai[:tree]
inchrome(D3Tree(tree))
is = interp_physical_state(s, sp, t)
frame = visualize(pomdp, is, r, tree=tree)

pngname = tempname()*".png"
write_to_png(frame, pngname)
run(`xdg-open $pngname`)
# gif = load(gifname)
# imshow(gif)
