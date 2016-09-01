using Multilane
using StatsBase
using JLD
using POMDPs
using POMDPToolbox
using POMCP
using GenerativeModels
using Plots


rng = MersenneTwister(58)

# behaviors = DiscreteBehaviorSet([normal, timid, aggressive], WeightVec(ones(3)))
# behaviors = DiscreteBehaviorSet(Multilane.NINE_BEHAVIORS, WeightVec(ones(9)))
behaviors = standard_uniform(1.0, correlated=true)

nb_lanes = 4
desired_lane_reward = 10.
rmodel = NoCrashRewardModel(desired_lane_reward*10., desired_lane_reward,2.5,nb_lanes)

pp = PhysicalParam(nb_lanes,lane_length=100.)

_discount = 1.0
nb_cars = 10
dmodel = NoCrashIDMMOBILModel(nb_cars, pp,
                          behaviors=behaviors,
                          lane_terminate=false,
                          vel_sigma=0.5,
                          p_appear=1.0)

pomdp = NoCrashPOMDP(dmodel, rmodel, 1.0)

sim = HistoryRecorder(rng=rng, capture_exception=false, max_steps=100)
# up = BehaviorRootUpdater(pomdp, WeightUpdateParams(smoothing=0.02))

up = AggressivenessUpdater(pomdp, 500, 0.05, WeightUpdateParams(smoothing=0.0, wrong_lane_factor=0.5), rng)

#=
solver = POMCPDPWSolver(
    eps=0.001,
    max_depth=20,
    c=50.0,
    tree_queries=2500,
    k_observation=4.0,
    alpha_observation=1/8,
    rollout_solver=SimpleSolver()
)
=#
# solver = BehaviorSolver(Multilane.UNIFORM_MEAN, false, rng)
solver = SimpleSolver()

policy = solve(solver, pomdp)

ips = MLPhysicalState(relaxed_initial_state(NoCrashMDP(dmodel,rmodel,1.0), 200, rng))
# id = DiscreteBehaviorBelief(ips, behaviors.models, [behaviors.weights.values for i in 1:length(ips.env_cars)])

nb_particles = 500
particles = [[rand(rng) for k in 1:nb_particles] for i in 1:length(ips.env_cars)]
weights = [ones(nb_particles) for i in 1:length(ips.env_cars)]
id = AggressivenessBelief(behaviors, ips, particles, weights)

println("starting simulation.")

simulate(sim, pomdp, policy, up, id)

@show length(sim.state_hist)

sh = sim.state_hist
bh = sim.belief_hist


# find out how many ids there are
max_id = 0
for s in sh
    max_id = max(max_id, maximum([c.id for c in s.env_cars]))
end

# T = 10
T = length(sh)

# columns are series
errors = Array(Float64, T, max_id-1)
fill!(errors, NaN)
stds = Array(Float64, T, max_id-1)
fill!(stds, NaN)
average_error = Array(Float64, T)

for i in 1:T
    b = bh[i]
    s = sh[i]
    total_error = 0.0
    mean_aggs = agg_means(b)
    ids = [c.id for c in s.env_cars[2:end]]
    stds[i,ids.-1] = agg_stds(b)[2:end]
    for j in 2:length(s.env_cars)
        true_agg = aggressiveness(b.gen, s.env_cars[j].behavior)
        err = abs(true_agg-mean_aggs[j])
        errors[i,s.env_cars[j].id-1] = err
        total_error += err
    end
    average_error[i] = total_error/length(s.env_cars)
end

@show mean(average_error)

# labels=collect(2:max_id)',
plot(average_error, linewidth=4, linecolor=:black, title="Aggressiveness Estimate Error", label="")
for i in 1:size(errors, 2)
    plot!(errors[:,i], label=(i-1))
end
plot!(stds, linestyle=:dash, labels="")
gui()

