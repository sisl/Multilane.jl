using Multilane
using GenerativeModels
using MCTS
using RobustMCTS
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

rsolver = RobustMCTSSolver(
    depth=20,
    c = 100.0,
    n_iterations=500,
    k_state=4.0,
    alpha_state=1/8,
    k_nature=4.0,
    alpha_nature=1/8,
    rollout_solver=SimpleSolver(),
    rollout_nature=StochasticBehaviorNoCrashMDP(mdp))


solver = RobustMLSolver(rsolver)
policy = solve(solver, mdp)

sim = HistoryRecorder(rng=rng, max_steps=100) # initialize a random number generator

simulate(sim, mdp, policy, s)

# check for crashes
for i in 1:length(sim.state_hist)-1
    if is_crash(mdp, sim.state_hist[i], sim.state_hist[i+1])
        println("""
        Crash!
        mdp = $mdp
        s = $(sim.state_hist[i])
        a = $(sim.action_hist[i])
        sp = $(sim.state_hist[i+1])
        """)
    end
    @test !is_crash(mdp, sim.state_hist[i], sim.state_hist[i+1])
end
