using ProgressMeter
using PmapProgressMeter
using POMDPToolbox
import Base: mean, std, repr, length

function test_run(problem::NoCrashMDP, initial_state::MLState, solver::Solver, rng::AbstractRNG=MersenneTwister())
    sim = POMDPToolbox.HistoryRecorder(rng=rng, max_steps=100)
    r = simulate(sim, problem, solve(solver,problem), initial_state)
    return sim
end

function test_run(problem::NoCrashPOMDP, bu, initial_state::MLState, solver::Solver, rng::AbstractRNG=MersenneTwister())
    sim = POMDPToolbox.HistoryRecorder(rng=rng, max_steps=100,initial_state=initial_state)
    r = simulate(sim, problem, solve(solver,problem), bu, create_belief(bu,initial_state))
    return sim
end

function run_simulations(problems::Vector, initial_states::Vector, solvers::Vector, bu=nothing; rng_offset::Int=100, parallel=true)
    # rewards = SharedArray(Float64, length(problems))
    N = length(problems)
    if parallel
        prog = ProgressMeter.Progress( N, dt=0.1, barlen=50, output=STDERR)
        if isa(bu,Void)
            sims = pmap(test_run,
                        prog,
                        problems,
                        initial_states,
                        solvers,
                        [MersenneTwister(j+rng_offset) for j in 1:N])
         else
             sims = pmap(test_run,
                         prog,
                         problems,
                         bu,
                         initial_states,
                         solvers,
                         [MersenneTwister(j+rng_offset) for j in 1:N])
         end
    else
        sims = Array(HistoryRecorder, length(problems))
        if isa(bu,Void)
          @showprogress for j in 1:length(problems)
              sims[j] = test_run(problems[j], initial_states[j], solvers[j], MersenneTwister(j+rng_offset))
          end
        else
          @showprogress for j in 1:length(problems)
              sims[j] = test_run(problems[j], bu[j], initial_states[j], solvers[j], MersenneTwister(j+rng_offset))
          end
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

function fill_stats!(stats::DataFrame, problems::Vector, sims::Vector)
    nb_sims = length(sims)

    # add new columns
    stats[:reward] = DataArray(Float64, nb_sims)
    stats[:brake_thresh] = Float64[p.rmodel.dangerous_brake_threshold for p in problems]
    stats[:lambda] = Float64[p.rmodel.cost_dangerous_brake/p.rmodel.reward_in_desired_lane for p in problems]
    stats[:nb_brakes] = DataArray(Int, nb_sims)
    stats[:steps_to_lane] = DataArray(Int, nb_sims)
    stats[:time_to_lane] = DataArray(Float64, nb_sims)
    stats[:steps_in_lane] = DataArray(Int, nb_sims)
    stats[:steps] = Int[length(s.action_hist) for s in sims]

    for i in 1:nb_sims
        r = 0.0
        dt = problems[i].dmodel.phys_param.dt
        steps_to_lane = Nullable{Int}()
        steps_in_lane = 0
        nb_brakes = 0

        for (k,a) in enumerate(sims[i].action_hist)
            s = sims[i].state_hist[k]
            sp = sims[i].state_hist[k+1]
            t = (k-1)*dt
            r += reward(problems[i], s, a, sp)

            nb_brakes += detect_braking(problems[i], s, sp)

            if s.env_cars[1].y == problems[i].rmodel.desired_lane
                if sp.env_cars[1].y == problems[i].rmodel.desired_lane
                    steps_in_lane += 1
                end
                if isnull(steps_to_lane)
                    steps_to_lane = Nullable{Int}(k-1)
                end
            end
        end

        if sims[i].state_hist[end].env_cars[1].y == problems[i].rmodel.desired_lane
            if isnull(steps_to_lane)
                steps_to_lane = Nullable{Int}(length(sims[i].state_hist)-1)
            end
        end

        stats[:reward][i] = r
        stats[:nb_brakes][i] = nb_brakes
        stats[:steps_to_lane][i] = isnull(steps_to_lane) ? NA : get(steps_to_lane)
        stats[:time_to_lane][i] = stats[:steps_to_lane][i]*dt
        stats[:steps_in_lane][i] = steps_in_lane
    end
    return stats
end

function evaluate(problems::Vector, initial_states::Vector, solvers::Dict{UTF8String, Solver}; rng_offset=0, parallel=true)
    nb_sims = length(problems)*length(initial_states)*length(solvers)
    all_problems = Array(Any, nb_sims)
    p_keys = UTF8String[randstring() for p in problems]
    all_initial = Array(Any, nb_sims)
    is_keys = UTF8String[randstring() for is in initial_states]
    all_solvers = Array(Any, nb_sims)
    stats = DataFrame(
        id=1:nb_sims,
        uuid=UInt128[Base.Random.uuid4() for i in 1:nb_sims],
        solver_key=DataArray(UTF8String,nb_sims),
        problem_key=DataArray(UTF8String,nb_sims),
        initial_key=DataArray(UTF8String,nb_sims),
        time=ones(nb_sims).*time(),
        )

    id = 0
    for p_i in 1:length(problems)
        for is_i in 1:length(initial_states)
            for solver_key in keys(solvers)
                id += 1
                stats[:solver_key][id] = solver_key
                all_solvers[id] = solvers[solver_key]
                stats[:problem_key][id] = p_keys[p_i]
                all_problems[id] = problems[p_i]
                stats[:initial_key][id] = is_keys[is_i]
                all_initial[id] = initial_states[is_i]
            end
        end
    end

    sims = run_simulations(all_problems, all_initial, all_solvers, nothing, rng_offset=rng_offset, parallel=parallel)
    fill_stats!(stats, all_problems, sims)
    return Dict{UTF8String, Any}(
        "nb_sims"=>nb_sims,
        "solvers"=>solvers,
        "problems"=>Dict{UTF8String,Any}([(p_keys[i], problems[i]) for i in 1:length(problems)]),
        "initial_states"=>Dict{UTF8String,Any}([(is_keys[i], initial_states[i]) for i in 1:length(initial_states)]),
        "stats"=>stats,
        "histories"=>sims
        )
end

function merge_results!(r1::Dict{UTF8String, Any}, r2::Dict{UTF8String, Any})
    r1["nb_sims"] += r2["nb_sims"]
    merge!(r1["solvers"], r2["solvers"])
    merge!(r1["problems"], r2["problems"])
    merge!(r1["initial_states"], r2["initial_states"])
    append!(r1["stats"], r2["stats"])
    r1["stats"][:id][end-r2["nb_sims"]+1:end] += r1["nb_sims"]
    append!(r1["histories"], r2["histories"])
    return r1
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
