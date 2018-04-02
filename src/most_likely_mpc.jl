struct MLMPCSolver <: Solver
    solver
end

function solve(sol::MLMPCSolver, p::NoCrashProblem)
    mdp = NoCrashMDP{typeof(p.rmodel), typeof(p.dmodel.behaviors)}(p.dmodel, p.rmodel, p.discount, p.throw) # make sure an MDP
    return MLMPCPolicy(solve(sol.solver, mdp))
end

struct MLMPCPolicy{P<:Policy} <: Policy
    planner::P
end

action_info(p::MLMPCPolicy, b) = action_info(p.planner, most_likely_state(b))

action(p::MLMPCPolicy, b) = first(action_info(p, b))

#=
mutable struct MLMPCSolver <: Solver
    solver
    updater::Nullable{Any}
end
function set_rng!(s::MLMPCSolver, rng::AbstractRNG)
    set_rng!(s.solver, rng)
    set_rng!(get(s.updater), rng)
end

mutable struct MLMPCAgent <: Policy
    updater::Updater
    previous_belief::Nullable{Any}
    policy::Policy
    previous_action::Nullable{MLAction}
end

function solve(solver::MLMPCSolver, problem::MLMDP, up=get(solver.updater, nothing))
    internal_problem = MLPOMDP{MLState, MLAction, MLPhysicalState, typeof(problem.dmodel), typeof(problem.rmodel)}(problem.dmodel, problem.rmodel, problem.discount)
    policy = solve(solver.solver, problem)
    if up == nothing
        up = POMDPs.updater(policy)
    else
        set_problem!(up, internal_problem)
    end
    return MLMPCAgent(up, nothing, policy, nothing)
end

function action(agent::MLMPCAgent, state::MLState)
    o = MLPhysicalState(state)
    if isnull(agent.previous_belief)
        belief = initialize_belief(agent.updater, ParticleGenerator(agent.policy.mdp, state))
    else
        belief = update(agent.updater, get(agent.previous_belief), get(agent.previous_action), o)
    end
    ml = most_likely_state(belief)
    a = action(agent.policy, ml)
    agent.previous_action = a
    agent.previous_belief = belief
    return a
end
=#
