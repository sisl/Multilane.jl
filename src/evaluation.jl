using ProgressMeter
using PmapProgressMeter
using POMDPToolbox

function test_run(problem::NoCrashMDP, initial_state::MLState, solver::Solver, rng::AbstractRNG=MersenneTwister())
    sim = POMDPToolbox.RolloutSimulator(rng=rng, max_steps=100)
    r = simulate(sim, problem, solve(solver,problem), initial_state)
    return r
end

function evaluate_performance(problems::Vector, initial_states::Vector, solver; rng_offset::Int=100, parallel=true)
    # rewards = SharedArray(Float64, length(problems))
    N = length(problems)
    if parallel
        prog = ProgressMeter.Progress( N, dt=0.1, barlen=50, output=STDERR)
        rewards = pmap(test_run,
                       prog,
                       problems,
                       initial_states,
                       [solver for i in 1:N],
                       [MersenneTwister(j+rng_offset) for j in 1:N]) 
    else
        rewards = Array(Float64, length(problems))
        @showprogress for j in 1:length(problems)
            rewards[j] = test_run(problems[j], initial_states[j], solver, MersenneTwister(j+rng_offset))
        end
    end
    for (i,r) in enumerate(rewards)
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
    
    return sdata(rewards)
end
