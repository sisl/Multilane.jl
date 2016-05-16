using ProgressMeter
using PmapProgressMeter
using POMDPToolbox
import Base: mean, std, repr, length


type NoCrashStat
  t_in_goal::Int
  nb_induced_brakes::Int
  reward::Real
end

type NoCrashStats
  stats::Array{NoCrashStat,1}
  rmodel::NoCrashRewardModel
end
length(ncs::NoCrashStats) = length(ncs.stats)

function mean(ncs::NoCrashStats)
  t = Int[stat.t_in_goal for stat in ncs.stats]
  b = Int[stat.nb_induced_brakes for stat in ncs.stats]
  r = Real[stat.reward for stat in ncs.stats]
  return mean(t), mean(b), mean(r)
end

function std(ncs::NoCrashStats)
  t = Int[stat.t_in_goal for stat in ncs.stats]
  b = Int[stat.nb_induced_brakes for stat in ncs.stats]
  r = Real[stat.reward for stat in ncs.stats]
  return std(t), std(b), std(r)
end

function ste(ncs::NoCrashStats)
  t,b,r = std(ncs)
  return t/sqrt(length(ncs)), b/sqrt(length(ncs)), r/sqrt(length(ncs))
end

function get_stats(problem::Union{NoCrashMDP,NoCrashPOMDP},sim::HistoryRecorder, r::Real)
  S = sim.state_hist
  A = sim.action_hist
  t_in_goal = 0
  for s in S
    if s.env_cars[1].y == problem.rmodel.desired_lane
      t_in_goal += 1
    end
  end
  # r = t_in_goal * reward_in_desired_lane - cost_dangerous_brake * nb_induced_brakes
  nb_induced_brakes = -(r - t_in_goal * problem.rmodel.reward_in_desired_lane) / problem.rmodel.cost_dangerous_brake
  return NoCrashStat(t_in_goal, nb_induced_brakes, r)
end

function test_run(problem::NoCrashMDP, initial_state::MLState, solver::Solver, rng::AbstractRNG=MersenneTwister())
    sim = POMDPToolbox.HistoryRecorder(rng=rng, max_steps=100)
    r = simulate(sim, problem, solve(solver,problem), initial_state)
    return get_stats(problem, sim, r)
end

function test_run(problem::NoCrashPOMDP, bu, initial_state::MLState, solver::Solver, rng::AbstractRNG=MersenneTwister())
    sim = POMDPToolbox.HistoryRecorder(rng=rng, max_steps=100,initial_state=initial_state)
    r = simulate(sim, problem, solve(solver,problem), bu, create_belief(bu,initial_state))
    return get_stats(problem, sim, r)
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
        rewards = Array(NoCrashStat, length(problems))
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

    # NOTE assumes same problem
    return NoCrashStats(sdata(rewards), problems[1].rmodel)
end
