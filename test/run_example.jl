using GenerativeModels
using MCTS
using POMDPs
using POMDPToolbox

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

rmodel = OriginalRewardModel(r_crash,
                             accel_cost,
                             decel_cost,
                             invalid_cost,
                             lineride_cost,
                             lanechange_cost)

dmodel = IDMMOBILModel(nb_cars, pp)

mdp = OriginalMDP(dmodel, rmodel, _discount, true);

# initialize the solver
# the hyper parameters in MCTS can be tricky to set properly
# n_iterations: the number of iterations that each search runs for
# depth: the depth of the tree (how far away from the current state the algorithm explores)
# exploration constant: this is how much weight to put into exploratory actions. 
# A good rule of thumb is to set the exploration constant to what you expect the upper bound on your average 
#  expected reward to be.

solver = MCTSSolver(n_iterations=100, depth=10, exploration_constant=1.0, enable_tree_vis=true)

# initialize the policy by passing in your problem and the solver
policy = solve(solver, mdp);

sim = HistoryRecorder(rng=MersenneTwister(9), max_steps=10) # initialize a random number generator

simulate(sim, mdp, policy, initial_state(mdp, sim.rng))
