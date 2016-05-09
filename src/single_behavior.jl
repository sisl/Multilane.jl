# Code for solving assuming that all other cars have a single identical behavior model

type SingleBehaviorSolver <: Solver
    inner_solver::Solver
    behavior::BehaviorModel
end

type SingleBehaviorPolicy <: Policy{MLState}
    inner_policy::Policy{MLState}
    behavior::BehaviorModel
end

function solve(solver::SingleBehaviorSolver, mdp::NoCrashMDP)
    single_behavior_mdp = deepcopy(mdp)
    single_behavior_mdp.dmodel.behaviors = BehaviorModel[solver.behavior]
    single_behavior_mdp.dmodel.behavior_probabilities = WeightVec([1.0])

    inner_policy = solve(solver.inner_solver, single_behavior_mdp)
    return SingleBehaviorPolicy(inner_policy, solver.behavior)
end

function action(p::SingleBehaviorPolicy, s::MLState, a::MLAction=MLAction())
    new_cars = Array(CarState, length(s.env_cars))
    for (i,c) in enumerate(s.env_cars)
        new_cars[i] = CarState(c.x, c.y, c.vel, c.lane_change, p.behavior, c.id)
    end
    return action(p.inner_policy, MLState(s.crashed, new_cars))
end
