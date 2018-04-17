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

Base.srand(p::MLMPCPolicy, s) = srand(p.planner, s)
action_info(p::MLMPCPolicy, b) = action_info(p.planner, most_likely_state(b))
action(p::MLMPCPolicy, b) = first(action_info(p, b))

struct MeanMPCSolver <: Solver
    solver
end

function solve(sol::MeanMPCSolver, p::NoCrashProblem)
    mdp = NoCrashMDP{typeof(p.rmodel), typeof(p.dmodel.behaviors)}(p.dmodel, p.rmodel, p.discount, p.throw) # make sure an MDP
    return MeanMPCPlanner(solve(sol.solver, mdp))
end

struct MeanMPCPlanner{P<:Policy} <: Policy
    planner::P
end

action_info(p::MeanMPCPlanner, b) = action_info(p.planner, mean(b))

action(p::MeanMPCPlanner, b) = first(action_info(p, b))


