using Multilane
using StatsBase
using JLD
using POMDPs
using POMDPToolbox
using Plots


rng = MersenneTwister(58)

behaviors = standard_uniform(1.0, correlation=0.75)

nb_lanes = 4
desired_lane_reward = 10.
rmodel = NoCrashRewardModel(desired_lane_reward*10., desired_lane_reward,4.0,nb_lanes)

pp = PhysicalParam(nb_lanes,lane_length=100.)

_discount = 1.0
nb_cars = 10
dmodel = NoCrashIDMMOBILModel(nb_cars, pp,
                          behaviors=behaviors,
                          lane_terminate=false,
                          vel_sigma=0.5,
                          p_appear=1.0)

pomdp = NoCrashPOMDP{typeof(rmodel), typeof(behaviors)}(dmodel, rmodel, 1.0, true)

sim = HistoryRecorder(rng=rng, show_progress=true, capture_exception=false, max_steps=100)
# up = BehaviorRootUpdater(pomdp, WeightUpdateParams(smoothing=0.02))

up = BehaviorParticleUpdater(pomdp, 10000, 0.01, 0.5, WeightUpdateParams(smoothing=0.0, wrong_lane_factor=0.5), rng)

# solver = BehaviorSolver(Multilane.UNIFORM_MEAN, false, rng)
solver = SimpleSolver()

policy = solve(solver, pomdp)

ips = MLPhysicalState(initial_state(pomdp, rng))
# id = DiscreteBehaviorBelief(ips, behaviors.models, [behaviors.weights.values for i in 1:length(ips.cars)])

id = ParticleGenerator(ips, behaviors)

println("starting simulation.")

hist = @time simulate(sim, pomdp, policy, up, id)

@show length(hist.state_hist)

sh = hist.state_hist
bh = hist.belief_hist


# find out how many ids there are
max_id = 0
for s in sh
    max_id = max(max_id, maximum([c.id for c in s.cars]))
end

# T = 10
T = length(sh)

# columns are series
errors = Array{IDMMOBILBehavior}(T, max_id-1)
fill!(errors, nan(IDMMOBILBehavior))
stds = Array{IDMMOBILBehavior}(T, max_id-1)
fill!(stds, nan(IDMMOBILBehavior))
average_error = Array{IDMMOBILBehavior}(T)

for i in 1:T
    b = bh[i]
    s = sh[i]
    total_error = zero(IDMMOBILBehavior)
    mean_params = param_means(b)
    ids = [c.id for c in s.cars[2:end]]
    stds[i,ids.-1] = param_stds(b)[2:end]
    for j in 2:length(s.cars)
        true_b = s.cars[j].behavior
        err = abs(true_b-mean_params[j])
        errors[i,s.cars[j].id-1] = err
        total_error += err
    end
    average_error[i] = total_error/length(s.cars)
end

@show mean(average_error)

plts = []
fields = Dict{Symbol, Vector{Symbol}}(
    :p_idm => [:a, :b, :T, :v0, :s0],
    :p_mobil => [:p, :b_safe, :a_thr]    
)

g = getfield

gen = behaviors
nf = IDMMOBILBehavior(gen.max_idm-gen.min_idm, gen.max_mobil-gen.min_mobil, 0) # normalization factor

pyplot()

for p in [:p_idm, :p_mobil]
    for f in fields[p]
        this_nf = g(g(nf,p),f)
        avg_err = [g(g(b,p),f) for b in average_error]./this_nf
        plot(avg_err, linewidth=4, linecolor=:black, title="normalized $p.$f", label="", ylim=(0,1))
        for i in 1:size(errors, 2)
            these_errors = [g(g(b,p),f) for b in errors[:,i]]./this_nf
            plot!(these_errors, label="")
            these_stds = [g(g(b,p),f) for b in stds[:,i]]./this_nf
            plot!(these_stds, linestyle=:dash, labels="")
        end
        push!(plts, plot!())
    end
end

plot(plts...)
gui()
