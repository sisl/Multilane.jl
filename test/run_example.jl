using GenerativeModels
using MCTS
using POMDPs

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

mdp = OriginalMDP(dmodel, rmodel, _discount);

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

rng = MersenneTwister(9) # initialize a random number generator
n_ep = 1
Rs = [Float64[] for _=1:n_ep]#zeros(n_ep)
nb_early_term = 0
histS = Array{MLState,1}[]
histA = Array{MLAction,1}[]
for j = 1:n_ep
    push!(histS,MLState[])
    push!(histA,MLAction[])
    s = initial_state(mdp,rng)
    rtot = 0.0
    disc = 1.0
    for i = 1:10
        # get the action from our SARSOP policy
        a = action(policy, s) # the QMDP action function returns the POMDP action not its index like the SARSOP action function
        #if j == n_ep
            push!(histS[j],s)
            push!(histA[j],a)
        #end
        # compute the reward
        r = reward(mdp, s, a)
        push!(Rs[j],r)
        rtot += disc*r
        if isterminal(mdp,s,a)
            nb_early_term += 1
            break
        end
        disc *= discount(mdp)
        print("\rEp:$j, t=$i")
        # transition the system state
        s = generate_s(mdp,s,a,rng)
    end
    #Rs[j] = rtot
    print("\rTotal discounted reward: $rtot\n")
end
