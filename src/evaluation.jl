import Base: mean, std, repr, length
import Iterators: take

function set_rng!(solver::Solver, rng::AbstractRNG)
    solver.rng = rng
end

function test_run(eval_problem::NoCrashMDP, initial_state::MLState, solver_problem::NoCrashMDP, solver::Solver, rng_seed::Integer, max_steps=10000)
    set_rng!(solver, MersenneTwister(rng_seed))
    sim = POMDPToolbox.HistoryRecorder(rng=MersenneTwister(rng_seed), max_steps=max_steps, capture_exception=false)
    terminal_problem = deepcopy(eval_problem)
    terminal_problem.dmodel.lane_terminate = true
    r = simulate(sim, terminal_problem, solve(solver,solver_problem), initial_state)
    return sim
end

function test_run_return_policy(eval_problem::NoCrashMDP, initial_state::MLState, solver_problem::NoCrashMDP, solver::Solver, rng_seed::Integer, max_steps=10000)
    set_rng!(solver, MersenneTwister(rng_seed))
    sim = POMDPToolbox.HistoryRecorder(rng=MersenneTwister(rng_seed), max_steps=max_steps)
    policy = solve(solver, solver_problem)
    terminal_problem = deepcopy(eval_problem)
    terminal_problem.dmodel.lane_terminate = true
    r = simulate(sim, terminal_problem, policy, initial_state)
    return sim, policy
end

function run_simulations(stats::DataFrame,
                         objects::Dict;
                         parallel=true,
                         max_steps=10000,
                         progress=true,
                         desc="Progress: ")

    nb_sims = nrow(stats)

    all_solvers = Array(Any, nb_sims)
    all_problems = Array(Any, nb_sims)
    all_solver_problems = Array(Any, nb_sims)
    all_initial = Array(Any, nb_sims)

    solvers = objects["solvers"]
    problems = objects["problems"]
    initial_states = objects["initial_states"]

    for i in 1:nrow(stats)
        all_solvers[i] = deepcopy(solvers[stats[i,:solver_key]])
        all_problems[i] = problems[stats[i,:problem_key]]
        all_solver_problems[i] = problems[stats[i,:solver_problem_key]]
        all_initial[i] = initial_states[stats[i,:initial_key]]
    end

    return run_simulations(all_problems,
                           all_initial,
                           all_solver_problems,
                           all_solvers,
                           stats[:rng_seed],
                           parallel=parallel,
                           max_steps=max_steps,
                           progress=progress,
                           desc=desc)
end

function run_simulations(eval_problems::AbstractVector,
                         initial_states::AbstractVector,
                         solver_problems::AbstractVector,
                         solvers::AbstractVector,
                         rng_seeds::AbstractVector=collect(1:length(eval_problems));
                         parallel=true,
                         max_steps=10000,
                         progress=true,
                         desc="Progress: ")

    N = length(eval_problems)
    prog = ProgressMeter.Progress( N, dt=0.1, barlen=30, output=STDERR, desc=desc)
    if parallel
        if progress
            sims = pmap(test_run,
                        prog,
                        eval_problems,
                        initial_states,
                        solver_problems,
                        solvers,
                        rng_seeds,
                        max_steps*ones(Int,N)
                        )
        else
            sims = pmap(test_run,
                        eval_problems,
                        initial_states,
                        solver_problems,
                        solvers,
                        rng_seeds,
                        max_steps*ones(Int,N)
                        )
        end
    else
        sims = Array(HistoryRecorder, length(eval_problems))
        if progress
            @showprogress for j in 1:length(eval_problems)
                sims[j] = test_run(eval_problems[j],
                                   initial_states[j],
                                   solver_problems[j],
                                   solvers[j],
                                   rng_seeds[j],
                                   max_steps)
            end
        else
            for j in 1:length(eval_problems)
                sims[j] = test_run(eval_problems[j],
                                   initial_states[j],
                                   solver_problems[j],
                                   solvers[j],
                                   rng_seeds[j],
                                   max_steps)
            end
        end
    end

    for (i,r) in enumerate(sims)
        if isa(r, RemoteException) || !isnull(r.exception)
            println("Exception in simulation $(i)!")
            println("===============================")
            if isa(r, RemoteException)
                Base.showerror(STDOUT, r.captured.ex)
                println()
                # @show r.captured.processed_bt
                Base.show_backtrace(STDOUT, r.captured.processed_bt)
                println()
            else
                Base.showerror(STDOUT, get(r.exception))
                println()
                Base.show_backtrace(STDOUT, get(r.backtrace))
                println()
            end
            println("===============================")
            println("")
        end
    end
    
    # return a vector of history recorders 
    return sims
end

function fill_stats!(stats::DataFrame, objects::Dict, sims::Vector;
                     metrics::AbstractVector=get(objects, "metrics", []))
    @assert nrow(stats) == length(sims)
    problems = objects["problems"]
    nb_sims = length(sims)
    eval_problems = [problems[key] for key in stats[:problem_key]]

    # add new columns
    stats[:reward] = DataArray(Float64, nb_sims)
    stats[:brake_thresh] = Float64[p.rmodel.dangerous_brake_threshold for p in eval_problems]
    stats[:lambda] = Float64[p.rmodel.cost_dangerous_brake/p.rmodel.reward_in_desired_lane for p in eval_problems]
    stats[:nb_brakes] = DataArray(Int, nb_sims)
    stats[:steps_to_lane] = DataArray(Int, nb_sims)
    stats[:time_to_lane] = DataArray(Float64, nb_sims)
    stats[:steps_in_lane] = DataArray(Int, nb_sims)
    stats[:steps] = Int[length(s.action_hist) for s in sims]
    stats[:crash] = DataArray(Bool, nb_sims)

    for m in metrics
        stats[key(m)] = DataArray(datatype(m), nb_sims)
    end

    for i in 1:nb_sims
        r = 0.0
        dt = eval_problems[i].dmodel.phys_param.dt
        steps_to_lane = Nullable{Int}()
        steps_in_lane = 0
        nb_brakes = 0
        crashed = false

        for (k,a) in enumerate(sims[i].action_hist)
            s = sims[i].state_hist[k]
            sp = sims[i].state_hist[k+1]
            t = (k-1)*dt
            r += reward(eval_problems[i], s, a, sp)

            nb_brakes += detect_braking(eval_problems[i], s, sp)

            if sp.cars[1].y == eval_problems[i].rmodel.desired_lane
                steps_in_lane += 1
            end
            if s.cars[1].y == eval_problems[i].rmodel.desired_lane
                if isnull(steps_to_lane)
                    steps_to_lane = Nullable{Int}(k-1)
                end
            end

            if is_crash(eval_problems[i], s, sp)
                crashed = true
            end
        end

        if sims[i].state_hist[end].cars[1].y == eval_problems[i].rmodel.desired_lane
            if isnull(steps_to_lane)
                steps_to_lane = Nullable{Int}(length(sims[i].state_hist)-1)
            end
        end

        stats[:reward][i] = r
        stats[:nb_brakes][i] = nb_brakes
        stats[:steps_to_lane][i] = isnull(steps_to_lane) ? NA : get(steps_to_lane)
        stats[:time_to_lane][i] = stats[:steps_to_lane][i]*dt
        stats[:steps_in_lane][i] = steps_in_lane
        stats[:crash][i] = crashed

        for m in metrics
            stats[key(m)][i] = calculate(m, eval_problems[i], sims[i])
        end
    end
    return stats
end

function assign_keys(problems::Vector, initial_states::Vector; rng=MersenneTwister(rand(UInt32)))
    p_keys = UTF8String[randstring(rng) for p in problems]
    is_keys = UTF8String[randstring(rng) for is in initial_states]
    return Dict{UTF8String, Any}(
        "eval_problems"=>Dict{UTF8String,Any}([(p_keys[i], problems[i]) for i in 1:length(problems)]),
        "initial_states"=>Dict{UTF8String,Any}([(is_keys[i], initial_states[i]) for i in 1:length(initial_states)]),
    )
end

function assign_keys(problems::AbstractVector; rng=MersenneTwister(rand(UInt32)))
    p_keys = UTF8String[randstring(rng) for p in problems]
    return Dict{UTF8String,Any}([(p_keys[i], problems[i]) for i in 1:length(problems)])
end


"""
Checks that all common keys have equal values in both dictionaries and adds keys not in common
"""
function careful_merge!(d1::Dict, d2::Dict)
    for k in keys(d2)
        if haskey(d1, k)
            @assert d1[k] == d2[k]
        else
            d1[k] = d2[k]
        end
    end
    return d1
end

function merge_results!{S1<:AbstractString, S2<:AbstractString}(r1::Dict{S1, Any}, r2::Dict{S2, Any}; careful=true)
    merge!(r1["solvers"], r2["solvers"])
    merge!(r1["problems"], r2["problems"])
    if careful
        careful_merge!(r1["initial_states"], r2["initial_states"])
    else
        merge!(r1["initial_states"], r2["initial_states"])
    end
    merge!(r1["behaviors"], r2["behaviors"])
    r1["param_table"] = join(r1["param_table"], r2["param_table"], on=names(r1["param_table"]), kind=:outer)
    if haskey(r1, "stats")
        # len = nrow(r1["stats"])
        append!(r1["stats"], r2["stats"])
        # r1["stats"][:id][end-len+1:end] += len
        r1["stats"][:id]=1:nrow(r1["stats"])
    else
        r1["stats"] = r2["stats"]
    end
    if haskey(r2, "histories")
        if haskey(r1, "histories")
            if r1["histories"] != nothing && r2["histories"] != nothing
                append!(r1["histories"], r2["histories"])
            end
        else
        r1["histories"] = r2["histories"]
        end
    end

    r1ips = r1["initial_physical_states"]
    if length(r2["initial_physical_states"]) > length(r1ips)
        r1["initial_physical_states"] = r2["initial_physical_states"]
    end
    for i in 1:min(length(r1ips), length(r2["initial_physical_states"]))
        @assert r1ips[i] == r2["initial_physical_states"][i]
    end

    merge_tests!(r1["tests"], r2["tests"])
    return r1
end


function rerun{S<:AbstractString}(results::Dict{S, Any}, id; enforce_match=true)
    stats = results["stats"]
    @assert stats[:id][id] == id
    problem = results["problems"][stats[:problem_key][id]]
    is = results["initial_states"][stats[:initial_key][id]]
    solver = results["solvers"][stats[:solver_key][id]]
    solver_problem = results["problems"][stats[:solver_problem_key][id]]
    rng_seed = stats[:rng_seed][id]
    if enforce_match
        steps = stats[:steps][id]
    else
        steps = 10000
    end
    sim, policy = test_run_return_policy(problem, is, solver_problem, solver, rng_seed, steps)
    r = sum([reward(problem, sim.state_hist[i], sim.action_hist[i], sim.state_hist[i+1]) for i in 1:length(sim.action_hist)])
    if enforce_match
        @assert r == stats[:reward][id]
    end
    return problem, sim, policy
end
