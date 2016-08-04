import Base: mean, std, repr, length
import Iterators: take

function test_run(eval_problem::NoCrashMDP, initial_state::MLState, soln_problem::NoCrashMDP, solver::Solver, rng_seed::Integer, max_steps=10000)
    if isa(solver, RandomSolver)
        solver.rng = MersenneTwister(rng_seed)
    end
    sim = POMDPToolbox.HistoryRecorder(rng=MersenneTwister(rng_seed), max_steps=max_steps)
    terminal_problem = deepcopy(eval_problem)
    terminal_problem.dmodel.lane_terminate = true
    r = simulate(sim, terminal_problem, solve(solver,soln_problem), initial_state)
    return sim
end

function test_run_return_policy(eval_problem::NoCrashMDP, initial_state::MLState, soln_problem::NoCrashMDP, solver::Solver, rng_seed::Integer, max_steps=10000)
    if isa(solver, RandomSolver)
        solver.rng = MersenneTwister(rng_seed)
    end
    sim = POMDPToolbox.HistoryRecorder(rng=MersenneTwister(rng_seed), max_steps=max_steps)
    policy = solve(solver, soln_problem)
    terminal_problem = deepcopy(eval_problem)
    terminal_problem.dmodel.lane_terminate = true
    r = simulate(sim, terminal_problem, policy, initial_state)
    return sim, policy
end

#=
function test_run(problem::NoCrashPOMDP, bu::Updater, initial_state::MLState, solver::Solver, rng_seed::Integer, max_steps=10000)
    # error("Not maintained: look over the code before using this")
    sim = POMDPToolbox.HistoryRecorder(rng=MersenneTwister(rng_seed), max_steps=max_steps,initial_state=initial_state)
    policy = solve(solver, problem)
    terminal_problem = deepcopy(problem)
    terminal_problem.dmodel.lane_terminate = true
    r = simulate(sim, problem, solve(solver,problem), bu, create_belief(bu,initial_state))
    return sim
end
=#

function run_simulations(eval_problems::AbstractVector,
                         initial_states::AbstractVector,
                         soln_problems::AbstractVector,
                         solvers::AbstractVector;
                         rng_seeds::AbstractVector=collect(1:length(eval_problems)),
                         parallel=true,
                         max_steps=10000,
                         desc="Progress: ")

    N = length(eval_problems)
    prog = ProgressMeter.Progress( N, dt=0.1, barlen=30, output=STDERR, desc=desc)
    if parallel
        sims = pmap(test_run,
                    prog,
                    eval_problems,
                    initial_states,
                    soln_problems,
                    solvers,
                    rng_seeds,
                    max_steps*ones(Int,N)
                    )
    else
        sims = Array(HistoryRecorder, length(eval_problems))
        @showprogress for j in 1:length(eval_problems)
            sims[j] = test_run(eval_problems[j], initial_states[j], solvers[j], rng_seeds[j], max_steps)
        end
    end

    for (i,r) in enumerate(sims)
        if isa(r, RemoteException)
            println("Exception in simulation $(i)!")
            println("===============================")
            println(r.captured.ex)
            # @show r.captured.processed_bt
            Base.show_backtrace(STDOUT, r.captured.processed_bt)
            println("===============================")
            println("")
        end
    end
    
    # return a vector of history recorders 
    return sims
end

function fill_stats!(stats::DataFrame, eval_problems::Vector, sims::Vector)
    nb_sims = length(sims)

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

            if sp.env_cars[1].y == eval_problems[i].rmodel.desired_lane
                steps_in_lane += 1
            end
            if s.env_cars[1].y == eval_problems[i].rmodel.desired_lane
                if isnull(steps_to_lane)
                    steps_to_lane = Nullable{Int}(k-1)
                end
            end

            if is_crash(eval_problems[i], s, sp)
                crashed = true
            end
        end

        if sims[i].state_hist[end].env_cars[1].y == eval_problems[i].rmodel.desired_lane
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

function evaluate(problem_keys::AbstractVector,
                  is_keys::AbstractVector,
                  solver_keys::AbstractVector,
                  objects::Dict{UTF8String, Any};
                  soln_problem_keys::AbstractVector=problem_keys,
                  N=length(initial_states),
                  rng_offset=0, parallel=true,
                  desc="Progress: ")

    @assert length(problem_keys) == length(soln_problem_keys)
    nb_sims = length(problem_keys)*min(N,length(is_keys))*length(solver_keys)
    all_problems = Array(Any, nb_sims)
    ep_keys = problem_keys
    all_soln_problems = Array(Any, nb_sims)
    sp_keys = soln_problem_keys
    all_initial = Array(Any, nb_sims)
    all_solvers = Array(Any, nb_sims)

    solvers = objects["solvers"]
    problems = objects["problems"]
    initial_states = objects["initial_states"]

    stats = DataFrame(
        id=1:nb_sims,
        uuid=UInt128[Base.Random.uuid4() for i in 1:nb_sims],
        solver_key=DataArray(UTF8String,nb_sims),
        eval_problem_key=DataArray(UTF8String,nb_sims),
        soln_problem_key=DataArray(UTF8String,nb_sims),
        initial_key=DataArray(UTF8String,nb_sims),
        rng_seed=DataArray(Int,nb_sims),
        time=ones(nb_sims).*time(),
        )

    id = 0
    for j in 1:length(problem_keys)
        ep_key = ep_keys[j]
        sp_key = sp_keys[j]
        for (is_i, is_key) in enumerate(take(is_keys, N))
            for solver_key in solver_keys
                id += 1
                stats[:solver_key][id] = solver_key
                all_solvers[id] = deepcopy(solvers[solver_key])
                stats[:eval_problem_key][id] = ep_key
                all_problems[id] = problems[ep_key]
                stats[:soln_problem_key][id] = sp_key
                all_soln_problems[id] = problems[sp_key]
                stats[:initial_key][id] = is_key
                all_initial[id] = initial_states[is_key]
                stats[:rng_seed][id] = is_i+rng_offset
            end
        end
    end

    sims = run_simulations(all_problems, all_initial, all_soln_problems, all_solvers, rng_seeds=stats[:rng_seed], parallel=parallel, desc=desc)
    fill_stats!(stats, all_problems, sims)
    return Dict{UTF8String, Any}(
        "solvers"=>solvers,
        "problems"=>problems,
        "initial_states"=>initial_states,
        "stats"=>stats,
        "histories"=>sims
        )
end



function merge_results!{S1<:AbstractString, S2<:AbstractString}(r1::Dict{S1, Any}, r2::Dict{S2, Any})
    merge!(r1["solvers"], r2["solvers"])
    merge!(r1["problems"], r2["problems"])
    merge!(r1["initial_states"], r2["initial_states"])
    if haskey(r1, "stats")
        len = nrow(r1["stats"])
        append!(r1["stats"], r2["stats"])
        r1["stats"][:id][end-len+1:end] += len
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
    return r1
end

function rerun{S<:AbstractString}(results::Dict{S, Any}, id; reward_assertion=true)
    stats = results["stats"]
    @assert stats[:id][id] == id
    problem = results["problems"][stats[:eval_problem_key][id]]
    is = results["initial_states"][stats[:initial_key][id]]
    solver = results["solvers"][stats[:solver_key][id]]
    soln_problem = results["soln_problems"][stats[:soln_problem_key][id]]
    rng_seed = stats[:rng_seed][id]
    steps = stats[:steps][id]
    sim, policy = test_run_return_policy(problem, is, solver, rng_seed, steps)
    r = sum([reward(problem, sim.state_hist[i], sim.action_hist[i], sim.state_hist[i+1]) for i in 1:length(sim.action_hist)])
    if reward_assertion
        @assert r == stats[:reward][id]
    end
    return problem, sim, policy
end


#=
function save_results(results, filename=string("results_", Dates.format(Dates.now(),"_u_d_HH_MM"), ".jld"))
    string_dict = Dict([(string(k),v) for (k,v) in results])
    save(filename, string_dict)
end

function load_results(filename)
    string_dict = load(filename)
    return Dict{Symbol, Any}([(symbol(k), v) for (k,v) in string_dict])
end
=#

#=
results::Dict{UTF8String, Any}
    - nb_sims::Int
    - solvers::Dict{UTF8String,Solver}
    - problems::Dict{UTF8String,Any}
    - initial_states::Dict{UTF8String,Any}
    - stats::DataFrame
    - histories::Vector{HistoryRecorder}
=#

#= evaluate
inputs: problems, initial conditions, solvers 
outputs: sim_dict
=#

#= Dataframe fields
    id::Int
    uuid::UInt128
    solver_key::UTF8String
    problem_key::UTF8String
    initial_key::UTF8String
    time::Float64

    # filled later
    reward::Float64
    brake_thresh::Float64
    lambda::Float64 # brake_cost/lane_reward
    nb_brakes::Int
    steps_to_lane::Int
    time_to_lane::Float64
    steps_in_lane::Int
    steps::Int
=#

# TODO: rng stuff?
