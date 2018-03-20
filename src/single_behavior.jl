# Code for solving assuming that all other cars have a single identical behavior model

mutable struct SingleBehaviorSolver <: Solver
    inner_solver::Solver
    behavior::BehaviorModel
end

mutable struct SingleBehaviorPolicy <: Policy
    inner_policy::Policy
    behavior::BehaviorModel
end

set_rng!(solver::SingleBehaviorSolver, rng::AbstractRNG) = set_rng!(solver.inner_solver, rng)

function solve(solver::SingleBehaviorSolver, mdp::NoCrashProblem)
    single_behavior_mdp = deepcopy(mdp)
    single_behavior_mdp.dmodel.behaviors = DiscreteBehaviorSet(BehaviorModel[solver.behavior], Weights([1.0]))

    inner_policy = solve(solver.inner_solver, single_behavior_mdp)
    return SingleBehaviorPolicy(inner_policy, solver.behavior)
end

function action(p::SingleBehaviorPolicy, s::Union{MLPhysicalState, MLState})
    as = actions(p.inner_policy.mdp, s, actions(p.inner_policy.mdp))
    if length(as) == 1
        return first(as)
    end
    return action(p.inner_policy, single_behavior_state(s, p.behavior))
end

action(p::SingleBehaviorPolicy, agg::AggressivenessBelief) = action(p, agg.physical)


function single_behavior_state(s::Union{MLState, MLPhysicalState}, behavior)
    new_cars = Vector{CarState}(length(s.cars))
    for (i,c) in enumerate(s.cars)
        new_cars[i] = CarState(c.x, c.y, c.vel, c.lane_change, behavior, c.id)
    end
    return MLState(s, new_cars)
end
