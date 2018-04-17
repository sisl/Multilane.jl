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
Base.srand(p::SingleBehaviorPolicy, s) = srand(p.inner_policy, s)

function solve(solver::SingleBehaviorSolver, mdp::NoCrashProblem)
    dmodel = NoCrashIDMMOBILModel(mdp.dmodel, DiscreteBehaviorSet(BehaviorModel[solver.behavior], Weights([1.0])))
    single_behavior_mdp = NoCrashMDP{typeof(mdp.rmodel), DiscreteBehaviorSet}(dmodel, mdp.rmodel, mdp.discount, mdp.throw)

    inner_policy = solve(solver.inner_solver, single_behavior_mdp)
    return SingleBehaviorPolicy(inner_policy, solver.behavior)
end

function action_info(p::SingleBehaviorPolicy, s::Union{MLPhysicalState, MLState})
    as = actions(p.inner_policy.mdp, s)
    if length(as) == 1
        return first(as), Dict(:tree_queries=>missing, :search_time_us=>missing)
    end
    return action_info(p.inner_policy, single_behavior_state(s, p.behavior))
end

action_info(p::SingleBehaviorPolicy, agg::AggressivenessBelief) = action_info(p, agg.physical)

action(p::SingleBehaviorPolicy, s) = first(action_info(p, s))

function single_behavior_state(s::Union{MLState, MLPhysicalState}, behavior)
    new_cars = Vector{CarState}(length(s.cars))
    for (i,c) in enumerate(s.cars)
        new_cars[i] = CarState(c.x, c.y, c.vel, c.lane_change, behavior, c.id)
    end
    return MLState(s, new_cars)
end
