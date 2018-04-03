
#push!(LOAD_PATH,joinpath("..","src"))

using Multilane
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

mdp = NoCrashMDP{typeof(rmodel), typeof(dmodel.behaviors)}(dmodel, rmodel, _discount, true);

rng = MersenneTwister(5)

s = initial_state(mdp::NoCrashMDP, rng)
# @show s.cars[1]
#visualize(mdp,s,MLAction(0,0))

policy = Multilane.BehaviorPolicy(mdp, Multilane.NORMAL, false, rng)

sim = HistoryRecorder(rng=rng, max_steps=100) # initialize a random number generator

hist = simulate(sim, mdp, policy, s)

# check for crashes
for i in 1:length(state_hist(hist))-1
    if is_crash(mdp, state_hist(hist)[i], state_hist(hist)[i+1])
        println("Crash:")
        println("mdp = $mdp\n")
        println("s = $(state_hist(hist)[i])\n")
        println("a = $(action_hist(hist)[i])\n")
        println("Saving gif...")
        f = write_tmp_gif(mdp, hist)
        println("gif written to $f")
    end
    @test !is_crash(mdp, state_hist(hist)[i], state_hist(hist)[i+1])
end

# for i in 1:length(sim.state_hist)-1
#     if is_crash(mdp, sim.state_hist[i], sim.state_hist[i+1])
#         visualize(mdp, sim.state_hist[i], sim.action_hist[i], sim.state_hist[i+1], two_frame_crash=true)
#         # println(repr(mdp))
#         # println(repr(sim.state_hist[i]))
#         println("Crash after step $i")
#         println("Chosen Action: $(sim.action_hist[i])")
#         println("Available actions:")
#         for a in actions(mdp, sim.state_hist[i], actions(mdp))
#             println(a)
#         end
#         println("Press Enter to continue.")
#         readline(STDIN)
#     end
# end

behaviors = standard_uniform(correlation=0.75)
dmodel = NoCrashIDMMOBILModel(dmodel, behaviors)
pomdp = NoCrashPOMDP{typeof(rmodel), typeof(dmodel.behaviors)}(dmodel, rmodel, _discount, true);

rng = MersenneTwister(5)

s = initial_state(mdp::NoCrashMDP, rng)

policy = Multilane.BehaviorPolicy(pomdp, Multilane.NORMAL, false, rng)

sim = HistoryRecorder(rng=rng, max_steps=100) # initialize a random number generator

wup = WeightUpdateParams(smoothing=0.0, wrong_lane_factor=0.5)
up = BehaviorParticleUpdater(pomdp, 5000, 0.1, 0.1, wup, MersenneTwister(50000))

# hist = simulate(sim, pomdp, policy, up, MLPhysicalState(s), s)
@time hist = simulate(sim, pomdp, policy, up, MLPhysicalState(s), s)
@show n_steps(hist)

# check for crashes
for i in 1:length(state_hist(hist))-1
    if is_crash(mdp, state_hist(hist)[i], state_hist(hist)[i+1])
        println("Crash:")
        println("mdp = $mdp\n")
        println("s = $(state_hist(hist)[i])\n")
        println("a = $(action_hist(hist)[i])\n")
        println("Saving gif...")
        f = write_tmp_gif(mdp, hist)
        println("gif written to $f")
    end
    @test !is_crash(mdp, state_hist(hist)[i], state_hist(hist)[i+1])
end

