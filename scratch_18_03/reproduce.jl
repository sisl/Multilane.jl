# 5 UCT exploration parameter
# 500 iterations per step
# 2500 POMCP iterations per step
# 500 simulations

using Multilane
using MCTS
using POMDPToolbox
using POMDPs
using Missings
using DataFrames
using CSV
using POMCPOW

@everywhere using Missings
@everywhere using Multilane
@everywhere using POMDPToolbox

@show N = 500
@show n_iters = 500
@show max_time = Inf
@show max_depth = 20
@show val = SimpleSolver()
alldata = DataFrame()

dpws = DPWSolver(depth=max_depth,
                 n_iterations=n_iters,
                 max_time=max_time,
                 exploration_constant=5.0,
                 k_state=4.0,
                 alpha_state=1/8,
                 enable_action_pw=false,
                 check_repeat_state=false,
                 estimate_value=RolloutEstimator(val)
                 # estimate_value=val
                )

wup = WeightUpdateParams(smoothing=0.0, wrong_lane_factor=0.5)

solvers = Dict{String, Solver}(
    "baseline" => SingleBehaviorSolver(dpws, Multilane.NORMAL),
    "omniscient" => dpws,
    "mlmpc" => MLMPCSolver(dpws),
    # "qmdp" => QBSolver(dpws),
    # "pftdpw" => begin
    #     m = 10
    #     wup = WeightUpdateParams(smoothing=0.0, wrong_lane_factor=0.5)
    #     rng = MersenneTwister(123)
    #     up = AggressivenessUpdater(nothing, m, 0.1, 0.1, wup, rng)
    #     ABMDPSolver(dpws, up)
    # end,
    # "pomcpow" => POMCPOWSolver(tree_queries=n_iters,
    #                            criterion=MaxUCB(5.0),
    #                            max_depth=max_depth,
    #                            max_time=max_time,
    #                            enable_action_pw=false,
    #                            k_observation=4.0,
    #                            alpha_observation=1/8,
    #                            estimate_value=FORollout(val),
    #                            # estimate_value=val,
    #                            check_repeat_obs=false,
    #                            node_sr_belief_updater=AggressivenessPOWFilter(wup)
    #                           )
)

# for lambda in 2.0.^(0:5)
for lambda in [0.0]
    @show lambda
    cor = 0.75

    behaviors = standard_uniform(correlation=cor)
    pp = PhysicalParam(4, lane_length=100.0)
    dmodel = NoCrashIDMMOBILModel(10, pp,
                                  behaviors=behaviors,
                                  p_appear=1.0,
                                  lane_terminate=false,
                                  max_dist=Inf
                                 )
    rmodel = NoCrashRewardModel(lambda, 1.0, 4.0, 4)
    # rmodel = SuccessReward(lambda=lambda)
    pomdp = NoCrashPOMDP{typeof(rmodel)}(dmodel, rmodel, 0.95, false)
    mdp = NoCrashMDP{typeof(rmodel)}(dmodel, rmodel, 0.95, false)

    problems = Dict{String, Any}(
        "baseline"=>mdp,
        "omniscient"=>mdp
    )
    solver_problems = Dict{String, Any}(
        "qmdp"=>mdp
    )

    for (k, solver) in solvers
        @show k
        p = get(problems, k, pomdp)
        sp = get(solver_problems, k, p)
        planner = solve(solver, sp)
        sim_problem = deepcopy(p)
        sim_problem.throw=true
        sim_problem.dmodel.lane_terminate=true

        sims = []

        for i in 1:N
            rng_seed = i+40000
            rng = MersenneTwister(rng_seed)
            is = initial_state(p, rng)
            ips = MLPhysicalState(is)

            metadata = Dict(:rng_seed=>rng_seed,
                            :lambda=>lambda,
                            :solver=>k,
                            :dt=>pp.dt,
                            :correlation=>convert(Float64, cor)
                       )   
            hr = HistoryRecorder(rng=rng, capture_exception=false)

            if p isa POMDP
                agg_up = AggressivenessUpdater(sim_problem, 500, 0.1, 0.1, WeightUpdateParams(smoothing=0.0, wrong_lane_factor=0.5), MersenneTwister(rng_seed+50000))
                push!(sims, Sim(sim_problem, planner, agg_up, ips, is,
                                simulator=hr,
                                metadata=metadata
                               ))
            else
                push!(sims, Sim(sim_problem, planner, is,
                                simulator=hr,
                                metadata=metadata
                               ))
            end
            @assert problem(last(sims)).throw
        end

        # data = run(sims) do sim, hist
        data = run_parallel(sims) do sim, hist

            if isnull(exception(hist))
                p = problem(sim)
                steps_in_lane = 0
                steps_to_lane = missing
                nb_brakes = 0
                crashed = false
                min_speed = Inf
                min_ego_speed = Inf
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

                    min_speed = min(minimum(c.vel for c in sp.cars), min_speed)
                    min_ego_speed = min(min_ego_speed, sp.cars[1].vel)
                end
                time_to_lane = steps_to_lane*p.dmodel.phys_param.dt
                distance = last(state_hist(hist)).x

                return [:n_steps=>n_steps(hist),
                        :mean_iterations=>mean(ai[:tree_queries] for ai in eachstep(hist, "ai")),
                        :mean_search_time=>1e-6*mean(ai[:search_time_us] for ai in eachstep(hist, "ai")),
                        :reward=>discounted_reward(hist),
                        :crashed=>crashed,
                        :steps_to_lane=>steps_to_lane,
                        :steps_in_lane=>steps_in_lane,
                        :nb_brakes=>nb_brakes,
                        :exception=>false,
                        :distance=>distance,
                        :mean_ego_speed=>distance/(n_steps(hist)*p.dmodel.phys_param.dt),
                        :min_speed=>min_speed,
                        :min_ego_speed=>min_ego_speed,
                        :terminal=>string(get(last(state_hist(hist)).terminal, missing)),
                        :init_n_cars=>length(first(state_hist(hist)).cars)
                       ]
            else
                warn("Error in Simulation")
                showerror(STDERR, get(exception(hist)))
                # show(STDERR, MIME("text/plain"), stacktrace(get(backtrace(hist))))
                return [:exception=>true,
                        :ex_type=>string(typeof(get(exception(hist))))
                       ]
            end
        end

        data[:time_to_lane] = data[:steps_to_lane].*p.dmodel.phys_param.dt
        @show extrema(data[:time_to_lane])
        @show mean(data[:time_to_lane])
        @show mean(data[:nb_brakes])
        @show extrema(data[:init_n_cars])
        @show mean(data[:init_n_cars])
        @show mean(data[:mean_search_time])
        @show mean(data[:reward])
        if minimum(data[:min_speed]) < 15.0
            @show minimum(data[:min_speed])
        end

        if isempty(alldata)
            alldata = data
        else
            alldata = vcat(alldata, data)
        end
    end
end

# @show alldata

datestring = Dates.format(now(), "E_d_u_HH_MM")
filename = Pkg.dir("Multilane", "data", "reproduction_"*datestring*".csv")
println("Writing data to $filename")
CSV.write(filename, alldata)
