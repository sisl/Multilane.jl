
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

s = initial_state(mdp::NoCrashMDP, rng)
# @show s.env_cars[1]
#visualize(mdp,s,MLAction(0,0))

policy = Multilane.BehaviorPolicy(mdp, Multilane.NORMAL, false, rng)

sim = HistoryRecorder(rng=rng, max_steps=100) # initialize a random number generator

simulate(sim, mdp, policy, s)

# check for crashes
for i in 1:length(sim.state_hist)-1
    if is_crash(mdp, sim.state_hist[i], sim.state_hist[i+1])
        println("Crash:")
        println("mdp = $mdp\n")
        println("s = $(sim.state_hist[i])\n")
        println("a = $(sim.action_hist[i])\n")
        println("Saving gif...")
        f = write_tmp_gif(mdp, sim)
        println("gif written to $f")
    end
    @test !is_crash(mdp, sim.state_hist[i], sim.state_hist[i+1])
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
