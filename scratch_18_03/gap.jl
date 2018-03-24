using Multilane
using MCTS
using POMDPToolbox
using POMDPs
# using POMCP
using Missings
using DataFrames
using CSV
using POMCPOW

@everywhere using Missings
@everywhere using Multilane
@everywhere using POMDPToolbox

using Gallium

@show N = 500
alldata = DataFrame()

dpws = DPWSolver(depth=20,
                 n_iterations=1_000_000,
                 max_time=1.0,
                 exploration_constant=10.0,
                 k_state=4.0,
                 alpha_state=1/8,
                 enable_action_pw=false,
                 check_repeat_state=false,
                 estimate_value=RolloutEstimator(SimpleSolver()))


solvers = Dict{String, Solver}(
    "baseline" => begin
        SingleBehaviorSolver(dpws, Multilane.NORMAL)
    end,
    "omniscient" => dpws,
    "mlmpc" => MLMPCSolver(dpws),
    "pftdpw" => begin
        m = 10
        wup = WeightUpdateParams(smoothing=0.0, wrong_lane_factor=0.5)
        rng = MersenneTwister(123)
        up = AggressivenessUpdater(nothing, m, 0.1, 0.1, wup, rng)
        ABMDPSolver(dpws, up)
    end,
    "pomcpow" => POMCPOWSolver(tree_queries=10_000_000,
                               criterion=MaxUCB(10.0),
                               max_depth=20,
                               max_time=1.0,
                               enable_action_pw=false,
                               k_state=4.0,
                               alpha_state=1/8,
                               estimate_value=FORollout(SimpleSolver()),
                               check_repeat_obs=false
                               node_sr_belief_updater=AggressivenessPOWFilter()
                              )
)



for lambda in 2.0.^(-2:4)
# for lambda in [1.0]
    @show lambda

    behaviors = standard_uniform(correlation=0.75)
    pp = PhysicalParam(4, lane_length=100.0)
    dmodel = NoCrashIDMMOBILModel(10, pp,
                                  behaviors=behaviors,
                                  p_appear=1.0,
                                  lane_terminate=true,
                                  max_dist=1000.0
                                 )
    rmodel = SuccessReward(lambda=lambda)
    pomdp = NoCrashPOMDP{typeof(rmodel)}(dmodel, rmodel, 0.95, false)
    mdp = NoCrashMDP{typeof(rmodel)}(dmodel, rmodel, 0.95, false)

    problems = Dict{String, Any}(
        "baseline"=>mdp,
        "omniscient"=>mdp
    )

    for (k, solver) in solvers
        @show k
        p = get(problems, k, pomdp)
        planner = solve(solver, p)
        sim_problem = deepcopy(p)
        sim_problem.throw=true

        sims = []

        for i in 1:N
            rng_seed = i+40000
            rng = MersenneTwister(rng_seed)
            is = initial_state(p, rng)
            ips = MLPhysicalState(is)

            metadata = Dict(:rng_seed=>rng_seed,
                            :lambda=>lambda,
                            :solver=>k,
                            :dt=>pp.dt
                       )   
            hr = HistoryRecorder(max_steps=100, rng=rng, capture_exception=false)

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
                        :terminal=>string(get(last(state_hist(hist)).terminal, missing))
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

        success = 100.0*sum(data[:terminal].=="lane")/N
        brakes = 100.0*sum(data[:nb_brakes].>=1)/N
        @printf("%% reaching:%5.1f; %% braking:%5.1f\n", success, brakes)

        @show extrema(data[:distance])
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
filename = Pkg.dir("Multilane", "data", "baseline_curve_"*datestring*".csv")
println("Writing data to $filename")
CSV.write(filename, alldata)
