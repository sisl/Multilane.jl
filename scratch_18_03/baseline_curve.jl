using Multilane
using MCTS
using JLD
using POMDPToolbox
using POMDPs
# using POMCP
using Missings
using DataFrames
using CSV

@everywhere using Missings
@everywhere using Multilane
@everywhere using POMDPToolbox

using Gallium

dpws = DPWSolver(depth=20,
                 n_iterations=1_000_000,
                 max_time=1.0,
                 exploration_constant=50.0,
                 k_state=4.0,
                 alpha_state=1/8,
                 estimate_value=RolloutEstimator(SimpleSolver()))


solver = SingleBehaviorSolver(dpws, Multilane.NORMAL)

@show N = 500
alldata = DataFrame()

for lambda in 2.^(0:5)
# for lambda in 2.^1

    @show lambda

    behaviors = standard_uniform(correlation=0.75)
    pp = PhysicalParam(4, lane_length=100.0)
    dmodel = NoCrashIDMMOBILModel(10, pp,
                                  behaviors=behaviors,
                                  p_appear=1.0,
                                  lane_terminate=true,
                                  max_dist=500.0
                                 )
    rmodel = SuccessReward(lambda=lambda,
                           target_lane=4,
                           brake_penalty_thresh=4.0
                          )
    pomdp = NoCrashPOMDP{typeof(rmodel)}(dmodel, rmodel, 0.95, false)
    mdp = NoCrashMDP{typeof(rmodel)}(dmodel, rmodel, 0.95, false)
    is = initial_state(pomdp, Base.GLOBAL_RNG)
    ips = MLPhysicalState(is)

    planner = solve(solver, mdp)

    sim_pomdp = deepcopy(pomdp)
    sim_pomdp.throw=true

    sims = []

    for i in 1:N
        rng_seed = i
        rng = MersenneTwister(rng_seed)
        agg_up = AggressivenessUpdater(sim_pomdp, 500, 0.1, 0.1, WeightUpdateParams(smoothing=0.0, wrong_lane_factor=0.5), MersenneTwister(rng_seed+50000))
        metadata = Dict(:rng_seed=>rng_seed,
                        :lambda=>lambda,
                        :solver=>"baseline",
                        :dt=>pp.dt
                   )   

        hr = HistoryRecorder(max_steps=100, rng=rng, capture_exception=true)
        push!(sims, Sim(sim_pomdp, planner, agg_up, ips, is,
                        simulator=hr,
                        metadata=metadata
                       ))
    end

    # data = run(sims) do sim, hist
    data = run_parallel(sims) do sim, hist

        if isnull(exception(hist))
            p = problem(sim)
            steps_in_lane = 0
            steps_to_lane = missing
            nb_brakes = 0
            crashed = false
            for (k,(s,sp)) in enumerate(eachstep(hist, "s,sp"))

                nb_brakes += detect_braking(p, s, sp)

                if sp.cars[1].y == p.rmodel.target_lane
                    steps_in_lane += 1
                end
                if sp.cars[1].y == p.rmodel.target_lane
                    if ismissing(steps_to_lane)
                        steps_to_lane = k
                    end
                end

                if is_crash(p, s, sp)
                    crashed = true
                end
            end
            time_to_lane = steps_to_lane*p.dmodel.phys_param.dt

            return [:n_steps=>n_steps(hist),
                    :mean_iterations=>mean(ai[:tree_queries] for ai in eachstep(hist, "ai")),
                    :reward=>discounted_reward(hist),
                    :crashed=>crashed,
                    :steps_to_lane=>steps_to_lane,
                    :steps_in_lane=>steps_in_lane,
                    :nb_brakes=>nb_brakes,
                    :exception=>false,
                    :terminal=>string(get(last(state_hist(hist)).terminal, missing))
                   ]
        else
            return [:exception=>true,
                    :ex_type=>string(typeof(get(exception(hist))))
                   ]
        end
    end

    if isempty(alldata)
        alldata = data
    else
        alldata = vcat(alldata, data)
    end
end

@show alldata

datestring = Dates.format(now(), "E_d_u_HH_MM")
filename = Pkg.dir("Multilane", "data", "baseline_curve_"*datestring*".csv")
println("Writing data to $filename")
CSV.write(filename, alldata)
