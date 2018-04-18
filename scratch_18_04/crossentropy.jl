using Multilane
using MCTS
using POMDPToolbox
using POMDPs
# using POMCP
using Missings
using DataFrames
using CSV
using POMCPOW
using Distributions

@everywhere using Missings
@everywhere using Multilane
@everywhere using POMDPToolbox

@show n_iters = 1000
@show max_time = Inf
@show max_depth = 40
@show val = SimpleSolver()
alldata = DataFrame()

function gen_sims(p, n, k, i)
    c = p[1]
    k_state = p[2]
    alpha_state = 1.0/p[3]
    dpws = DPWSolver(depth=max_depth,
                     n_iterations=n_iters,
                     max_time=max_time,
                     exploration_constant=c,
                     k_state=k_state,
                     alpha_state=alpha_state,
                     enable_action_pw=false,
                     check_repeat_state=false,
                     estimate_value=RolloutEstimator(val)
                     # estimate_value=val
                    )

    cor = 1.0
    lambda = 1.0

    behaviors = standard_uniform(correlation=cor)
    pp = PhysicalParam(4, lane_length=100.0)
    dmodel = NoCrashIDMMOBILModel(10, pp,
                                  behaviors=behaviors,
                                  p_appear=1.0,
                                  lane_terminate=true,
                                  max_dist=1000.0
                                 )
    rmodel = SuccessReward(lambda=lambda)
    mdp = NoCrashMDP{typeof(rmodel), typeof(behaviors)}(dmodel, rmodel, 0.95, false)

    sims = []

    for j in 1:n
        rng_seed = j + 40000*i*k
        planner = deepcopy(solve(dpws, mdp))
        srand(planner, rng_seed+50000)
        rng = MersenneTwister(rng_seed + 60000)
        is = initial_state(mdp, rng)
        ips = MLPhysicalState(is)

        sim = Sim(deepcopy(mdp),
                  deepcopy(planner),
                  rng=rng,
                  max_steps=100,
                  metadata=Dict(:i=>i, :k=>k)
                 )
        push!(sims, sim)
    end

    return sims
end

function make_updater(cor, problem, rng_seed)
    wup = WeightUpdateParams(smoothing=0.0, wrong_lane_factor=0.05)
    if cor >= 0.5
        return AggressivenessUpdater(problem, 2000, 0.05, 0.1, wup, MersenneTwister(rng_seed+50000))
    else
        return BehaviorParticleUpdater(problem, 5000, 0.05, 0.2, wup, MersenneTwister(rng_seed+50000))
    end
end

# params are c, k_state, alpha_state
start_mean = [2.0, 4.0, 8.0]
start_cov = diagm([10.0^2, 10.0^2, 30.0^2])
d = MvNormal(start_mean, start_cov)
K = 60 # 60 # number of parameter samples
n = 100 # 100 # number of evaluation simulations
m = 15  # 15 # number of elite samples
max_iters = 100

for i in 1:max_iters
    sims = []
    params = Vector{Vector{Float64}}(K)
    print("creating $K simulation sets")
    for k in 1:K
        p = rand(d)
        p[1] = max(0.0, p[1])
        p[2] = max(1.0, p[2])
        p[3] = max(2.0, p[3])
        params[k] = p
        k_sims = gen_sims(p, n, k, i)
        print(".")
        append!(sims, k_sims)
    end
    println()
    results = run_parallel(sims)
    # results = run(sims)
    combined = by(results, :k) do df
        DataFrame(mean_reward=mean(df[:reward]))
    end
    @show mean(combined[:mean_reward])
    order = sortperm(combined[:mean_reward])
    elite = params[combined[:k][order[K-m+1:end]]]
    elite_matrix = Matrix{Float64}(length(start_mean), m)
    for k in 1:m
        elite_matrix[:,k] = elite[k]
    end
    try
        d = fit(typeof(d), elite_matrix)
    catch ex
        if ex isa Base.LinAlg.PosDefException
            println("pos def exception")
            d = fit(typeof(d), elite_matrix += 0.01*randn(size(elite_matrix)))
        else
            rethrow(ex)
        end
    end
    println("iteration $i")
    @show mean(d)
    @show det(cov(d))
    @show ev = eigvals(cov(d))
    for j in 1:length(ev)
        @show eigvecs(cov(d))[:,j]
    end
end
